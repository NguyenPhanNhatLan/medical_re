# datasets/base_dataset.py
import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset

class BaseREDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer,     # chỉ nhận tokenizer cho nhẹ, nếu lấy encoder, sẽ bê hết hơn 400k tham số qua --> nặng bộ nhớ              
        max_length: int = 256,
        label2id: Dict[str, int] = None,
        entity_type2id: Dict[str, int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

        with open(json_path, "r", encoding="utf-8") as f:
            self.data: List[Dict] = json.load(f)

        # build label map nếu chưa có
        if label2id is None:
            labels = sorted(list({x["relation"] for x in self.data}))
            self.label2id = {l: i for i, l in enumerate(labels)}
        else:
            self.label2id = label2id
            
        if entity_type2id is None:
            types = set()
            for x in self.data:
                if "subj" in x and "type" in x["subj"]:
                    types.add(x["subj"]["type"])
                if "obj" in x and "type" in x["obj"]:
                    types.add(x["obj"]["type"])
            
        self.with_end_tags = False 

    def __len__(self):
        return len(self.data)

    @staticmethod
    def insert_entity_markers(text: str, e1: Dict, e2: Dict, with_end_tags: bool = True) -> str:
        # Gom tất cả điểm chèn cần thiết
        insertions = []
        
        # Entity 1
        insertions.append((e1["start"], "<e1>"))
        if with_end_tags:
            insertions.append((e1["end"], "</e1>"))
            
        # Entity 2
        insertions.append((e2["start"], "<e2>"))
        if with_end_tags:
            insertions.append((e2["end"], "</e2>"))

        # Sắp xếp:
        # 1. Vị trí (index) giảm dần (từ cuối về đầu để không làm lệch index phía trước)
        # 2. Nếu cùng vị trí, thẻ ĐÓNG (</e1>) phải đứng TRƯỚC thẻ MỞ (<e2>) để bao bọc đúng logic (nếu cần lồng nhau)
        #    hoặc tùy thuộc vào mong muốn của bạn.
        #    Ở đây ta sort: index giảm dần. 
        insertions.sort(key=lambda x: x[0], reverse=True)

        out = text
        for idx, tag in insertions:
            # Chèn tag vào vị trí idx
            out = out[:idx] + tag + out[idx:]
            
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
        e1 = item.get("entity_1", item.get("subj"))
        e2 = item.get("entity_2", item.get("obj"))
        
        if e1 is None or e2 is None:
            raise ValueError(f"Dữ liệu tại index {idx} thiếu thông tin entity.")

        marked_text = self.insert_entity_markers(
            item["text"], e1, e2, with_end_tags=self.with_end_tags
        )
        enc = self.tokenize(marked_text)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label_id": self.label2id[item["relation"]],
        }
