# datasets/base_dataset.py
# -*- coding: utf-8 -*-

import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


class BaseREDataset(Dataset):
    """
    Base dataset for Relation Extraction
    - Load json
    - Insert entity markers
    - Tokenize
    """

    def __init__(
        self,
        json_path: str,
        encoder,                    
        max_length: int = 256,
        label2id: Dict[str, int] = None,
    ):
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer
        self.max_length = max_length

        with open(json_path, "r", encoding="utf-8") as f:
            self.data: List[Dict] = json.load(f)

        # build label map nếu chưa có
        if label2id is None:
            labels = sorted({x["relation"] for x in self.data})
            self.label2id = {l: i for i, l in enumerate(labels)}
        else:
            self.label2id = label2id

    def __len__(self):
        return len(self.data)

    # -------- marker utils --------
    @staticmethod
    def insert_entity_markers(text: str, e1: Dict, e2: Dict, with_end_tags: bool = True) -> str:
        """
        - with_end_tags=True  -> R-BERT style: <e1>...</e1>, <e2>...</e2>
        - with_end_tags=False -> BERT-ES style: chỉ <e1>, <e2>
        """
        if with_end_tags:
            spans = [
                (e1["start"], e1["end"], "<e1>", "</e1>"),
                (e2["start"], e2["end"], "<e2>", "</e2>"),
            ]
            spans.sort(key=lambda x: x[0], reverse=True)

            out = text
            for s, e, ltag, rtag in spans:
                out = out[:s] + ltag + out[s:e] + rtag + out[e:]
            return out
        else:
            spans = [
                (e1["start"], "<e1>"),
                (e2["start"], "<e2>"),
            ]
            spans.sort(key=lambda x: x[0], reverse=True)

            out = text
            for s, tag in spans:
                out = out[:s] + tag + out[s:]
            return out

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }

    def get_common_fields(self, idx: int) -> Dict:
        item = self.data[idx]

        marked_text = self.insert_entity_markers(
            item["text"], item["entity_1"], item["entity_2"], with_end_tags=self.with_end_tags
        )
        enc = self.tokenize(marked_text)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label_id": self.label2id[item["relation"]],
        }
