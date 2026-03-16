from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm


BatchType = Tuple[torch.Tensor, ...]


@dataclass
class TrainHistory:
    train_loss_epoch: List[float]
    val_loss_epoch: List[float]
    val_accuracy_epoch: List[float]
    val_precision_epoch: List[float]
    val_recall_epoch: List[float]
    val_f1_epoch: List[float]
    best_f1: float
    best_epoch: int


class MyTrain:
    """
    Notebook-friendly trainer for HERANet.

    Design goals:
    - keep the original HERANet forward mechanism intact
    - keep original train.txt behavior (AdamW + ReduceLROnPlateau + CE + early stopping)
    - remove logging/config plumbing
    - allow optional augmentation function injection from augmentation.txt
    - allow batches of either:
        (input_ids, lengths, labels)
      or
        (input_ids, lengths, labels, domain_ids, meta)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_grad_norm: float = 5.0,
        epochs: int = 20,
        patience: int = 5,
        scheduler_patience: int = 2,
        scheduler_factor: float = 0.5,
        class_weights: Optional[List[float]] = None,
        augmentation_fn: Optional[Callable] = None,
        use_augmentation: bool = True,
        save_path: Optional[str] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.patience = patience
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.augmentation_fn = augmentation_fn
        self.use_augmentation = use_augmentation
        self.save_path = save_path

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
        )

        if class_weights is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        self.train_loss_epoch: List[float] = []
        self.val_loss_epoch: List[float] = []
        self.val_accuracy_epoch: List[float] = []
        self.val_precision_epoch: List[float] = []
        self.val_recall_epoch: List[float] = []
        self.val_f1_epoch: List[float] = []
        self.best_f1: float = -1.0
        self.best_epoch: int = -1
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def _unpack_batch(
        self,
        batch: BatchType,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(batch) == 3:
            input_ids, lengths, labels = batch
            domain_ids, meta = None, None
        elif len(batch) == 5:
            input_ids, lengths, labels, domain_ids, meta = batch
        else:
            raise ValueError(
                "Expected batch to have length 3 or 5: "
                "(input_ids, lengths, labels) or (input_ids, lengths, labels, domain_ids, meta)."
            )

        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)

        if domain_ids is not None:
            domain_ids = torch.as_tensor(domain_ids, dtype=torch.long, device=self.device)

        if meta is not None:
            meta = torch.as_tensor(meta, dtype=torch.float32, device=self.device)

        return input_ids, lengths, labels, domain_ids, meta

    def _maybe_apply_augmentation(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_augmentation or self.augmentation_fn is None:
            return input_ids, lengths, labels

        # original augmentation.txt style is easier to keep if tensors are detached first
        aug_out = self.augmentation_fn(
            input_ids.detach().clone(),
            lengths.detach().clone(),
            labels.detach().clone(),
            current_epoch=epoch,
            total_epochs=self.epochs,
        )

        if not isinstance(aug_out, (tuple, list)) or len(aug_out) != 3:
            raise ValueError("augmentation_fn must return (input_ids, lengths, labels).")

        aug_input_ids, aug_lengths, aug_labels = aug_out
        aug_input_ids = torch.as_tensor(aug_input_ids, dtype=torch.long, device=self.device)
        aug_lengths = torch.as_tensor(aug_lengths, dtype=torch.long, device=self.device)
        aug_labels = torch.as_tensor(aug_labels, dtype=torch.long, device=self.device)
        return aug_input_ids, aug_lengths, aug_labels

    def _forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        domain_ids: Optional[torch.Tensor],
        meta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.model(input_ids, lengths, domain_ids=domain_ids, meta=meta)

    def train_one_epoch(self, train_loader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        total_examples = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{self.epochs}", leave=False)
        for batch in pbar:
            input_ids, lengths, labels, domain_ids, meta = self._unpack_batch(batch)
            input_ids, lengths, labels = self._maybe_apply_augmentation(input_ids, lengths, labels, epoch)

            self.optimizer.zero_grad()
            logits = self._forward(input_ids, lengths, domain_ids, meta)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            total_examples += bs
            pbar.set_postfix(loss=running_loss / max(total_examples, 1))

        avg_loss = running_loss / max(total_examples, 1)
        self.train_loss_epoch.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        y_true: List[int] = []
        y_pred: List[int] = []

        pbar = tqdm(val_loader, desc="Valid", leave=False)
        for batch in pbar:
            input_ids, lengths, labels, domain_ids, meta = self._unpack_batch(batch)
            logits = self._forward(input_ids, lengths, domain_ids, meta)
            loss = self.criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            bs = labels.size(0)

            total_loss += loss.item() * bs
            total_examples += bs
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            pbar.set_postfix(loss=total_loss / max(total_examples, 1))

        avg_loss = total_loss / max(total_examples, 1)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        self.val_loss_epoch.append(avg_loss)
        self.val_accuracy_epoch.append(float(accuracy))
        self.val_precision_epoch.append(float(precision))
        self.val_recall_epoch.append(float(recall))
        self.val_f1_epoch.append(float(f1))

        return avg_loss, float(accuracy), float(precision), float(recall), float(f1)

    def _save_best(self):
        if self.save_path is None or self.best_state_dict is None:
            return
        torch.save(self.best_state_dict, self.save_path)

    def fit(self, train_loader, val_loader=None) -> TrainHistory:
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)

            if val_loader is None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch [{epoch}/{self.epochs}] train_loss: {train_loss:.4f} lr: {current_lr:.6f}")
                continue

            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate(val_loader)
            self.scheduler.step(val_f1)

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{self.epochs}] "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"acc: {val_acc:.4f} | "
                f"prec: {val_prec:.4f} | "
                f"rec: {val_rec:.4f} | "
                f"f1: {val_f1:.4f} | "
                f"lr: {current_lr:.6f}"
            )

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_epoch = epoch
                patience_counter = 0
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if val_loader is not None and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            print(f"Best epoch: {self.best_epoch}, best macro F1: {self.best_f1:.4f}")

        return TrainHistory(
            train_loss_epoch=self.train_loss_epoch,
            val_loss_epoch=self.val_loss_epoch,
            val_accuracy_epoch=self.val_accuracy_epoch,
            val_precision_epoch=self.val_precision_epoch,
            val_recall_epoch=self.val_recall_epoch,
            val_f1_epoch=self.val_f1_epoch,
            best_f1=self.best_f1,
            best_epoch=self.best_epoch,
        )
