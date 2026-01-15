# datasets/rbert_dataset.py
# -*- coding: utf-8 -*-

import torch
from datasets.base_dataset import BaseREDataset


class RBERTDataset(BaseREDataset):
    """
    Dataset cho R-BERT
    - tạo e1_mask, e2_mask để mean pooling
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_end_tags = True  # R-BERT style

    def _find_first(self, ids: torch.Tensor, token_id: int) -> int:
        pos = (ids == token_id).nonzero(as_tuple=False)
        return int(pos[0].item()) if pos.numel() > 0 else 0

    def build_span_masks(self, input_ids: torch.Tensor):
        tok = self.tokenizer

        e1_id = tok.convert_tokens_to_ids("<e1>")
        e1e_id = tok.convert_tokens_to_ids("</e1>")
        e2_id = tok.convert_tokens_to_ids("<e2>")
        e2e_id = tok.convert_tokens_to_ids("</e2>")

        L = input_ids.size(0)

        e1s = self._find_first(input_ids, e1_id)
        e1e = self._find_first(input_ids, e1e_id)
        e2s = self._find_first(input_ids, e2_id)
        e2e = self._find_first(input_ids, e2e_id)

        e1_mask = torch.zeros(L, dtype=torch.long)
        e2_mask = torch.zeros(L, dtype=torch.long)

        if e1e > e1s + 1:
            e1_mask[e1s + 1 : e1e] = 1
        if e2e > e2s + 1:
            e2_mask[e2s + 1 : e2e] = 1

        return e1_mask, e2_mask

    def __getitem__(self, idx):
        out = self.get_common_fields(idx)

        e1_mask, e2_mask = self.build_span_masks(out["input_ids"])
        out["e1_mask"] = e1_mask
        out["e2_mask"] = e2_mask

        return out
