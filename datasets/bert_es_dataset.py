# datasets/bert_es_dataset.py

import torch
from datasets.base_dataset import BaseREDataset
from datasets.utils import find_entity_positions

class BERTESDataset(BaseREDataset):
    def __init__(self, *args, **kwargs):  # <--- SỬA LẠI TÊN HÀM CHO ĐÚNG
        super().__init__(*args, **kwargs)
        self.with_end_tags = False  # BERT-ES style
        
        self.e1_id = self.tokenizer.convert_tokens_to_ids("<e1>")
        self.e2_id = self.tokenizer.convert_tokens_to_ids("<e2>")

    def __getitem__(self, idx):
        out = self.get_common_fields(idx)
        
        if not isinstance(out["input_ids"], torch.Tensor):
             out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)

        e1_pos, e2_pos = find_entity_positions(
            out["input_ids"], 
            self.e1_id, 
            self.e2_id
        )

        out["e1_pos"] = e1_pos
        out["e2_pos"] = e2_pos

        return out
