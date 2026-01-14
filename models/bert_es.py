import torch
import torch.nn as nn

class BERTSE(nn.Module):
    def __init__(self, encoder, hidden_size: int, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        """
        e1_pos, e2_pos: Tensor/List[(start_idx, end_idx)]
        """
        # 1. Encode
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = enc_out.hidden_states   # [B, L, H]
        cls_vec = enc_out.cls            # [B, H]

        # 2. Entity Start vectors
        e1_vecs = []
        e2_vecs = []
        B = input_ids.size(0)

        for i in range(B):
            s1 = e1_pos[i][0]
            s2 = e2_pos[i][0]

            # token sau <e1>, <e2>
            v1 = hidden[i, s1 + 1]
            v2 = hidden[i, s2 + 1]

            e1_vecs.append(v1)
            e2_vecs.append(v2)

        e1_vecs = torch.stack(e1_vecs)  # [B, H]
        e2_vecs = torch.stack(e2_vecs)  # [B, H]

        # 3. Classify
        x = torch.cat([cls_vec, e1_vecs, e2_vecs], dim=-1)
        logits = self.classifier(x)

        return logits
