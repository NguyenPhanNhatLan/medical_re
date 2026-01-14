import torch
import torch.nn as nn

class RBERT(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        # 1. Encode
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = enc_out.hidden_states   # [B, L, H]
        cls_vec = enc_out.cls            # [B, H]

        # 2. R-BERT pooling
        e1_vecs, e2_vecs = [], []
        B = input_ids.size(0)

        for i in range(B):
            s1, e1 = e1_pos[i]
            s2, e2 = e2_pos[i]

            # safety check
            v1 = hidden[i, s1+1:e1].mean(dim=0) if s1 + 1 < e1 else hidden[i, s1]
            v2 = hidden[i, s2+1:e2].mean(dim=0) if s2 + 1 < e2 else hidden[i, s2]

            e1_vecs.append(v1)
            e2_vecs.append(v2)

        e1_vecs = torch.stack(e1_vecs)
        e2_vecs = torch.stack(e2_vecs)

        # 3. Classify
        x = torch.cat([cls_vec, e1_vecs, e2_vecs], dim=-1)
        logits = self.classifier(x)

        return logits
