import torch
import torch.nn as nn

class RBERT(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        outputs = self.encoder(input_ids, attention_mask)
        hidden = outputs.last_hidden_state   # [B, L, H]

        cls_vec = hidden[:, 0, :]             # [B, H]

        e1_vecs, e2_vecs = [], []
        for i in range(input_ids.size(0)):
            s1, e1 = e1_pos[i]
            s2, e2 = e2_pos[i]

            v1 = hidden[i, s1+1:e1].mean(dim=0)
            v2 = hidden[i, s2+1:e2].mean(dim=0)

            e1_vecs.append(v1)
            e2_vecs.append(v2)

        e1_vecs = torch.stack(e1_vecs)
        e2_vecs = torch.stack(e2_vecs)

        x = torch.cat([cls_vec, e1_vecs, e2_vecs], dim=-1)
        logits = self.classifier(x)

        return logits
