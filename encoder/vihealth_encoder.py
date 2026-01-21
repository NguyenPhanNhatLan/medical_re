# models/vihealth_encoder.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


@dataclass
class EncoderOutput:
    hidden_states: torch.Tensor
    cls: torch.Tensor


class ViHealthBERTEncoder(nn.Module):
    """
    - Load tokenizer + model
    - Add entity marker tokens (<e1>, </e1>, <e2>, </e2>)
    - Provide forward() => EncoderOutput
    - Provide helper methods to compute:
        - span mean pooling (for R-BERT)
        - entity start vectors (for BERT-ES)
    """
    def __init__(
        self,
        model_name: str = "demdecuong/vihealthbert-base-word",
        add_entity_markers: bool = True,
        use_fast_tokenizer: bool = True,
    ):
        super().__init__()
        self.model_name = model_name

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

        self.hidden_size = self.model.config.hidden_size

        # cache token ids for quick lookup
        self._e1_id = self.tokenizer.convert_tokens_to_ids("<e1>")
        self._e2_id = self.tokenizer.convert_tokens_to_ids("<e2>")

    # ---------- core encode ----------
    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        **kwargs: Any
    ) -> EncoderOutput:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hs = out.last_hidden_state              # [B, L, H]
        cls = hs[:, 0, :]                       # [B, H]
        return EncoderOutput(hidden_states=hs, cls=cls)

    # ---------- helpers for heads ----------
    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor,    # [B, L, H]
        mask: torch.Tensor,            # [B, L] 0/1
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Mean pooling over a token span using a 0/1 mask.
        Return: [B, H]
        """
        m = mask.to(hidden_states.dtype).unsqueeze(-1)  # [B, L, 1]
        summed = (hidden_states * m).sum(dim=1)         # [B, H]
        denom = m.sum(dim=1).clamp_min(eps)             # [B, 1]
        return summed / denom

    @staticmethod
    def gather_positions(
        hidden_states: torch.Tensor,    # [B, L, H]
        positions: torch.Tensor         # [B]
    ) -> torch.Tensor:
        """
        Gather hidden vector at specified token index for each sample.
        Return: [B, H]
        """
        B, L, H = hidden_states.shape
        idx = positions.view(B, 1, 1).expand(B, 1, H)   # [B, 1, H]
        return hidden_states.gather(dim=1, index=idx).squeeze(1)

    def find_entity_start_positions(
        self,
        input_ids: torch.Tensor,  # [B, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optional utility: if your pipeline inserts <e1> and <e2> in text,
        you can auto-find their positions by searching input_ids.
        Return: e1_pos [B], e2_pos [B]
        """
        # input_ids == token_id -> [B, L] bool
        e1_mask = (input_ids == self._e1_id)
        e2_mask = (input_ids == self._e2_id)

        # argmax returns first True position if exists, else 0
        e1_pos = e1_mask.int().argmax(dim=1)
        e2_pos = e2_mask.int().argmax(dim=1)

        return e1_pos, e2_pos
        
    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)
        self.model.save_pretrained(save_directory)
