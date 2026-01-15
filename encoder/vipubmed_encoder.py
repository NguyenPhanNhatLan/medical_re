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
    """
    hidden_states: [B, L, H]
    cls:          [B, H]
    """
    hidden_states: torch.Tensor
    cls: torch.Tensor


class ViPubmedDeBERTaEncoder(nn.Module):
    """
    Encoder wrapper for ViPubmedDeBERTa.
    Safe for entity-aware RE and dual-encoder fusion.
    """

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

        # ---- Load tokenizer & model ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast_tokenizer,
        )
        self.model = AutoModel.from_pretrained(model_name)

        # ---- Add entity markers ----
        if add_entity_markers:
            added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": SPECIAL_TOKENS}
            )
            if added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))

        # ---- Gradient checkpointing (optional) ----
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.hidden_size = self.model.config.hidden_size

        # Cache token ids
        self._e1_id = self.tokenizer.convert_tokens_to_ids("<e1>")
        self._e2_id = self.tokenizer.convert_tokens_to_ids("<e2>")

    # ---------- core encode ----------
    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        **kwargs: Any
    ) -> EncoderOutput:

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,  # DeBERTa does not use segment embeddings
            **kwargs
        )

        hidden_states = out.last_hidden_state   # [B, L, H]
        cls = hidden_states[:, 0, :]            # [B, H]

        return EncoderOutput(hidden_states=hidden_states, cls=cls)

    # ---------- helpers ----------
    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor,    # [B, L, H]
        mask: torch.Tensor,            # [B, L]
        eps: float = 1e-8
    ) -> torch.Tensor:
        m = mask.to(hidden_states.dtype).unsqueeze(-1)
        summed = (hidden_states * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(eps)
        return summed / denom

    @staticmethod
    def gather_positions(
        hidden_states: torch.Tensor,    # [B, L, H]
        positions: torch.Tensor         # [B]
    ) -> torch.Tensor:
        B, L, H = hidden_states.shape
        idx = positions.view(B, 1, 1).expand(B, 1, H)
        return hidden_states.gather(dim=1, index=idx).squeeze(1)

    def find_entity_start_positions(
        self,
        input_ids: torch.Tensor,  # [B, L]
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
