import torch
import torch.nn as nn
from typing import Tuple
from transformers import AutoTokenizer, AutoModel

ENTITY_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

class PhoBERTEncoder(nn.Module):
    def __init__(self, model_name: str = "vinai/phobert-base", use_fast_tokenizer: bool = False):
        super().__init__()

        # Tokenizer & backbone
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast_tokenizer
        )
        self.backbone = AutoModel.from_pretrained(model_name)

        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ENTITY_TOKENS}
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.backbone.config.hidden_size
        self._e1_id = self.tokenizer.convert_tokens_to_ids("<e1>")
        self._e2_id = self.tokenizer.convert_tokens_to_ids("<e2>")

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        **kwargs
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state      # [B, L, H]
        cls_vector = hidden_states[:, 0, :]            # [B, H]

        return {
            "hidden_states": hidden_states,
            "cls": cls_vector
        }

    @staticmethod
    def gather_positions(
        hidden_states: torch.Tensor,   # [B, L, H]
        positions: torch.Tensor        # [B]
    ) -> torch.Tensor:
        """
        Lấy vector tại vị trí token xác định (BERT-ES)
        """
        B, _, H = hidden_states.size()
        index = positions.view(B, 1, 1).expand(B, 1, H)
        return hidden_states.gather(dim=1, index=index).squeeze(1)

    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor,   # [B, L, H]
        span_mask: torch.Tensor,       # [B, L]
        eps: float = 1e-9
    ) -> torch.Tensor:
        """
        Mean pooling theo span mask (R-BERT)
        """
        m = span_mask.unsqueeze(-1).type_as(hidden_states)
        summed = (hidden_states * m).sum(dim=1)
        count = m.sum(dim=1).clamp(min=eps)
        return summed / count

    def locate_entity_markers(
        self,
        input_ids: torch.Tensor        # [B, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tìm vị trí <e1>, <e2> trong input_ids (BERT-ES)
        """
        e1_pos = (input_ids == self._e1_id).int().argmax(dim=1)
        e2_pos = (input_ids == self._e2_id).int().argmax(dim=1)
        return e1_pos, e2_pos
