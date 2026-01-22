# models/vipubmed_encoder.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

@dataclass
class EncoderOutput:
    hidden_states: torch.Tensor
    cls: torch.Tensor
    # Nếu muốn dùng attentions/all_hidden_states thì phải khai báo thêm ở đây, 
    # nhưng với bài toán RE này thì không cần, nên ta sẽ bỏ nó lúc return.

class ViPubmedDeBERTaEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "manhtt-079/vipubmed-deberta-base",
        add_entity_markers: bool = True,
        use_fast_tokenizer: bool = True,
        gradient_checkpointing: bool = False,
        strict_entity_check: bool = False,
        tokenizer: Optional[AutoTokenizer] = None  # <--- THÊM THAM SỐ NÀY
    ):
        super().__init__()

        self.model_name = model_name
        self.strict_entity_check = strict_entity_check
        
        # 1. Ưu tiên dùng tokenizer truyền vào từ bên ngoài (để đồng bộ ID)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=use_fast_tokenizer,
            )
            
        self.model = AutoModel.from_pretrained(model_name)

        # 2. Thêm token và resize embedding
        if add_entity_markers:
            # Thêm nếu chưa có
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": SPECIAL_TOKENS}
            )
        
        # Luôn resize model nếu kích thước không khớp
        if self.model.config.vocab_size != len(self.tokenizer):
            self.model.resize_token_embeddings(len(self.tokenizer))

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.hidden_size = self.model.config.hidden_size

        self._e1_id = self.tokenizer.convert_tokens_to_ids("<e1>")
        self._e2_id = self.tokenizer.convert_tokens_to_ids("<e2>")

    def forward(
        self,
        input_ids: torch.Tensor,      
        attention_mask: torch.Tensor, 
        output_attentions: bool = False,      # <--- SỬA LẠI TYPE HINT
        output_hidden_states: bool = False,   # <--- SỬA LẠI TYPE HINT
        **kwargs
    ) -> EncoderOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        hidden_states = out.last_hidden_state
        cls = hidden_states[:, 0, :]   
        
        # <--- SỬA LẠI RETURN: Chỉ trả về những gì EncoderOutput định nghĩa
        return EncoderOutput(hidden_states=hidden_states, cls=cls)

    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        m = mask.to(hidden_states.dtype).unsqueeze(-1)
        summed = (hidden_states * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(eps)
        return summed / denom

    @staticmethod
    def gather_positions(
        hidden_states: torch.Tensor,
        positions: torch.Tensor         
    ) -> torch.Tensor:
        B, L, H = hidden_states.shape
        positions = positions.clamp(0, L - 1)
        idx = positions.view(B, 1, 1).expand(B, 1, H)
        return hidden_states.gather(dim=1, index=idx).squeeze(1)

    def find_entity_start_positions(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        e1_mask = (input_ids == self._e1_id)
        e2_mask = (input_ids == self._e2_id)

        missing_e1 = ~e1_mask.any(dim=1)
        missing_e2 = ~e2_mask.any(dim=1)

        if missing_e1.any() or missing_e2.any():
            msg = (
                f"Missing entity markers: "
                f"<e1>: {missing_e1.sum().item()}, "
                f"<e2>: {missing_e2.sum().item()}"
            )
            if self.strict_entity_check:
                raise ValueError(msg)
            # else:
            #     print(f"[Warning] {msg}")

        e1_pos = e1_mask.int().argmax(dim=1)
        e2_pos = e2_mask.int().argmax(dim=1)

        return e1_pos, e2_pos

    # ---------- save / load ----------
    def save_pretrained(self, save_dir: str):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, load_dir: str, **kwargs):
        return cls(model_name=load_dir, **kwargs)
