# datasets/bert_es_dataset.py
# -*- coding: utf-8 -*-

import torch
from datasets.base_dataset import BaseREDataset


class BERTESDataset(BaseREDataset):
    """
    Dataset cho BERT-ES
    - lấy vị trí <e1> và <e2>
    """
    def init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_end_tags = False  # BERT-ES style

    def __getitem__(self, idx):
        out = self.get_common_fields(idx)

        input_ids = out["input_ids"].unsqueeze(0)  # [1, L]
        e1_pos, e2_pos = self.encoder.find_entity_start_positions(input_ids)

        out["e1_pos"] = e1_pos.squeeze(0)
        out["e2_pos"] = e2_pos.squeeze(0)

        return out
