# datasets/bert_es_dataset.py
# -*- coding: utf-8 -*-

import torch
from datasets.base_dataset import BaseREDataset

class BERTESDataset(BaseREDataset):
    def __init__(self, json_path, encoder, max_length=256, label2id=None):
        # 1. Gọi init của lớp cha để load data và label2id
        super().__init__(
            json_path=json_path,
            encoder=encoder,
            max_length=max_length,
            label2id=label2id,
        )
        # 2. Ghi đè flag sau khi lớp cha đã khởi tạo
        self.with_end_tags = False  # BERT-ES chỉ dùng <e1> không dùng </e1>

    def __getitem__(self, idx):
        # Lấy các trường chung (input_ids, attention_mask, label_id)
        # Hàm này bên Base sẽ dùng self.with_end_tags đã khai báo ở trên
        out = self.get_common_fields(idx)

        # Thêm batch dim để tìm vị trí token (encoder yêu cầu [B, L])
        input_ids = out["input_ids"].unsqueeze(0)  

        # Tìm vị trí token <e1> và <e2>
        e1_pos, e2_pos = self.encoder.find_entity_start_positions(input_ids)

        # Loại bỏ batch dim [1] -> [] (scalar)
        out["e1_pos"] = e1_pos.squeeze()
        out["e2_pos"] = e2_pos.squeeze()

        return out