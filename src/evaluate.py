from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm


class MyEvaluate:
    """
    Minimal notebook-friendly evaluator for HERANet.

    Supports batches of either:
      (input_ids, lengths, labels)
    or
      (input_ids, lengths, labels, domain_ids, meta)
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None, criterion: Optional[nn.Module] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def _unpack_batch(self, batch):
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

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, object]:
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        y_true: List[int] = []
        y_pred: List[int] = []

        pbar = tqdm(data_loader, desc="Evaluate", leave=False)
        for batch in pbar:
            input_ids, lengths, labels, domain_ids, meta = self._unpack_batch(batch)
            logits = self.model(input_ids, lengths, domain_ids=domain_ids, meta=meta)
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

        result = {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
            "y_true": y_true,
            "y_pred": y_pred,
        }
        return result

    def print_summary(self, result: Dict[str, object]) -> None:
        print("=" * 60)
        print(f"loss           : {result['loss']:.4f}")
        print(f"accuracy       : {result['accuracy']:.4f}")
        print(f"precision_macro: {result['precision_macro']:.4f}")
        print(f"recall_macro   : {result['recall_macro']:.4f}")
        print(f"f1_macro       : {result['f1_macro']:.4f}")
        print("-" * 60)
        print("confusion_matrix")
        print(result["confusion_matrix"])
        print("-" * 60)
        print("classification_report")
        print(result["classification_report"])
        print("=" * 60)
