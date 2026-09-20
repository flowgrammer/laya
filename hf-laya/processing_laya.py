"""``LayaProcessor`` — packs ``(state, questions)`` into HF-native tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from transformers.processing_utils import ProcessorMixin

from .sequences import QTYPES, pack, render_options


class LayaProcessor(ProcessorMixin):
    """HF ``ProcessorMixin`` wrapping a tokenizer with typed-decision packing."""

    attributes: list[str] = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    max_len: int = 512
    head_max_len: int = 192

    def __init__(self, tokenizer, max_len: int | None = None, head_max_len: int | None = None) -> None:
        super().__init__(tokenizer)
        if max_len is not None:
            self.max_len = max_len
        if head_max_len is not None:
            self.head_max_len = head_max_len

    def __call__(
        self,
        state: str | Mapping | list,
        questions: Iterable[Mapping[str, Any]],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError(f"return_tensors must be 'pt', got {return_tensors!r}")

        questions = list(questions)
        packed = [pack(self.tokenizer, state, q, self.max_len, self.head_max_len) for q in questions]

        for q, (_, markers) in zip(questions, packed):
            if len(markers) < len(render_options(q)):
                raise ValueError(
                    f"question {q['id']!r} options exceed head_max_len={self.head_max_len}"
                )

        n = len(packed)
        seq_len = max(len(ids) for ids, _ in packed)
        k_max = max(len(m) for _, m in packed)

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((n, seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
        marker_pos = torch.zeros((n, k_max), dtype=torch.long)
        marker_mask = torch.zeros((n, k_max), dtype=torch.bool)

        for i, (ids, markers) in enumerate(packed):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            marker_pos[i, : len(markers)] = torch.tensor(markers, dtype=torch.long)
            marker_mask[i, : len(markers)] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": torch.tensor([QTYPES[q["kind"]] for q in questions], dtype=torch.long),
        }
