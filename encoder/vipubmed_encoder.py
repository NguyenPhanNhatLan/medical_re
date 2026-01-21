# models/vipubmeddeberta_encoder.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


@dataclass
class EncoderOutput:
    hidden_states: torch.Tensor
    cls: torch.Tensor


class ViPubmedDeBERTaEncoder(nn.Module):
    def __init__(
        self,
       model_name: str = "manhtt-079/vipubmed-deberta-base",
        add_entity_markers: bool = True,
        use_fast_tokenizer: bool = True,
        gradient_checkpointing: bool = False,
        strict_entity_check: bool = False,  
    ):
        super().__init__()

        self.model_name = model_name
        self.strict_entity_check = strict_entity_check
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast_tokenizer,
        )
        self.model = AutoModel.from_pretrained(model_name)

        if add_entity_markers:
            added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": SPECIAL_TOKENS}
            )
            if added > 0:
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
        output_attentions: False,
        output_hidden_states: False,
        **kwargs) -> EncoderOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        hidden_states = out.last_hidden_state
        cls = hidden_states[:, 0, :]   
        if output_hidden_states:
            all_hidden_states=out.hidden_states
        else: None
        
        if output_attentions:
            attentions=out.attentions
        else: None
        return EncoderOutput(hidden_states=hidden_states, cls=cls, attentions=attentions, all_hidden_states=all_hidden_states)

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
            else:
                print(f"[Warning] {msg}")

        e1_pos = e1_mask.int().argmax(dim=1)
        e2_pos = e2_mask.int().argmax(dim=1)

        return e1_pos, e2_pos

    # ---------- save / load ----------
    def save_pretrained(self, save_dir: str):
        """
        Save BOTH model and tokenizer.
        MUST be used instead of torch.save(state_dict).
        """
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, load_dir: str, **kwargs):
        """
        Reload encoder safely with tokenizer + model aligned.
        """
        return cls(model_name=load_dir, **kwargs)
