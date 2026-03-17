from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------
# Utility: masked softmax
# (1) pad 같이 의미 없는 것들을 제외하고 가중치를 확률로 맞추기 위해서
# -----------------------------
def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    scores = scores.masked_fill(~mask.bool(), float('-inf'))
    return torch.softmax(scores, dim=dim)

# -----------------------------
# Attention pooling 
# (1) Linear층들이 각 토큰에서 “중요도 판단에 필요한 특징”을 추출하고
# (2) 그 특징을 기반으로 attention score를 만들고
# (3) score로 가중합해서 최종 pooled representation을 만든다.
# -----------------------------
class AttentivePooling(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.proj(x))
        scores = self.v(u).squeeze(-1)
        attn = masked_softmax(scores, mask, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled

# -----------------------------
# BiLSTM Encoder
# (1) LSTM으로 문장을 양 방향으로 인코딩
# (2) 인코딩 한 문장을 attention pooling
# -----------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True, batch_first=True)
        self.out_dim = hidden_size * 2
        self.pool = AttentivePooling(self.out_dim, hidden=self.out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        max_len = out.size(1)
        device = out.device
        idxs = torch.arange(max_len, device=device).unsqueeze(0)
        mask = idxs < lengths.unsqueeze(1)
        pooled = self.pool(out, mask)
        return pooled, mask

# -----------------------------
# Hierarchical BiLSTM
# (1) 문서를 chunk 단위로 나눔
# (2) 각 chunk를 token LSTM에 넣어서 각각의 벡터로 만들음
# (3) 각각의 벡터를 시퀀스로 연결하여 한번 더 LSTM에 넣음
# -----------------------------
class HierBiLSTM(nn.Module):
    def __init__(self, embedding: nn.Embedding, chunk_size: int = 256, chunk_stride: int = 224,
                 token_hidden: int = 128, token_layers: int = 1, sent_hidden: int = 128,
                 sent_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.token_rnn = BiLSTMEncoder(embedding.embedding_dim, hidden_size=token_hidden,
                                       num_layers=token_layers, dropout=dropout)
        self.sent_rnn = BiLSTMEncoder(self.token_rnn.out_dim, hidden_size=sent_hidden,
                                      num_layers=sent_layers, dropout=dropout)
        self.out_dim = self.sent_rnn.out_dim
        self.dropout = nn.Dropout(dropout)

    def _make_chunks(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        starts = [list(range(0, int(l.item()) if l.item()>0 else 0, self.chunk_stride)) for l in lengths]
        for i, l in enumerate(lengths.tolist()):
            if l <= 0: continue
            last_start = max(0, l - self.chunk_size)
            if len(starts[i]) == 0 or starts[i][-1] != last_start:
                starts[i].append(last_start)
        S = max(len(s) for s in starts) if starts else 1
        chunks = input_ids.new_full((B, S, self.chunk_size), fill_value=self.embedding.padding_idx)
        chunk_lens = input_ids.new_zeros((B, S))
        for b in range(B):
            offs = starts[b]
            for s_idx in range(len(offs)):
                st = offs[s_idx]
                ed = min(st + self.chunk_size, lengths[b].item())
                cur_len = max(0, ed - st)
                if cur_len > 0:
                    chunks[b, s_idx, :cur_len] = input_ids[b, st:ed]
                chunk_lens[b, s_idx] = cur_len
        return chunks, chunk_lens

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build chunks
        chunks, chunk_lens = self._make_chunks(input_ids, lengths)  # [B, S, C], [B, S]
        B, S, C = chunks.shape

        # Token-level encoding with SAFE handling of zero-length chunks
        emb = self.embedding(chunks.view(B * S, C))                 # [B*S, C, E]
        token_lengths = chunk_lens.view(B * S)
        valid_tok = token_lengths > 0

        chunk_repr = emb.new_zeros((B * S, self.token_rnn.out_dim))
        if valid_tok.any():
            rep_valid, _ = self.token_rnn(emb[valid_tok], token_lengths[valid_tok])
            chunk_repr[valid_tok] = rep_valid
        chunk_repr = chunk_repr.view(B, S, -1)                      # [B, S, Dtok]

        # Sentence-level (chunk sequence) encoding with SAFE handling of zero-sentence docs
        sent_lengths = (chunk_lens > 0).sum(dim=1)                  # [B]
        valid_doc = sent_lengths > 0

        doc_repr = chunk_repr.new_zeros((B, self.sent_rnn.out_dim))
        sent_mask = None
        if valid_doc.any():
            doc_valid, sent_mask_valid = self.sent_rnn(chunk_repr[valid_doc], sent_lengths[valid_doc])
            doc_repr[valid_doc] = doc_valid
            # synthesize a simple mask for all docs: True up to length, else False
            max_S = S
            idxs = torch.arange(max_S, device=chunk_repr.device).unsqueeze(0)
            full_mask = idxs < sent_lengths.unsqueeze(1)
            sent_mask = full_mask
        else:
            # No valid docs in this batch (edge case) -> keep zeros
            max_S = S
            idxs = torch.arange(max_S, device=chunk_repr.device).unsqueeze(0)
            sent_mask = idxs < sent_lengths.unsqueeze(1)

        doc_repr = self.dropout(doc_repr)
        return doc_repr, sent_mask

# -----------------------------
# Final unified class: BiLSTM
# (1) domain data는 문서의 유형을 알려줘서 학습을 돕는 것
# (2) meta data는 문서 길이, 댓글 수 등을 알려줘서 학습을 돕는 것
# -----------------------------
class HERANet(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 2, embedding_dim: int = 128,
                 padding_idx: int = 0, chunk_size: int = 256, chunk_stride: int = 224,
                 token_hidden: int = 128, token_layers: int = 1, sent_hidden: int = 128,
                 sent_layers: int = 1, dropout: float = 0.2, meta_dim: int = 0, meta_out: int = 64,
                 use_domain_embedding: bool = True, num_domains: int = 3, domain_emb_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.encoder = HierBiLSTM(self.embedding, chunk_size, chunk_stride,
                                  token_hidden, token_layers, sent_hidden, sent_layers, dropout)
        rep_dim = self.encoder.out_dim
        # Meta branch (store out dim for zero-padding at runtime)
        if meta_dim > 0:
            self.meta_mlp = nn.Sequential(nn.Linear(meta_dim, meta_out), nn.ReLU(), nn.Dropout(dropout))
            self.meta_out = meta_out
            rep_dim += meta_out
        else:
            self.meta_mlp = None
            self.meta_out = 0
        # Domain branch (store emb dim for zero-padding at runtime)
        if use_domain_embedding:
            self.domain_emb = nn.Embedding(num_domains, domain_emb_dim)
            self.domain_emb_dim = domain_emb_dim
            rep_dim += domain_emb_dim
        else:
            self.domain_emb = None
            self.domain_emb_dim = 0
        self.classifier = nn.Linear(rep_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor,
                domain_ids: Optional[torch.Tensor] = None, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = input_ids.size(0)
        doc_repr, _ = self.encoder(input_ids, lengths)
        reps = [doc_repr]
        # Always align dims with zeros when inputs are missing
        if self.domain_emb is not None:
            if domain_ids is not None:
                reps.append(self.domain_emb(domain_ids))
            else:
                reps.append(doc_repr.new_zeros((B, self.domain_emb_dim)))
        if self.meta_mlp is not None:
            if meta is not None:
                reps.append(self.meta_mlp(meta))
            else:
                reps.append(doc_repr.new_zeros((B, self.meta_out)))
        h = torch.cat(reps, dim=-1)
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits
