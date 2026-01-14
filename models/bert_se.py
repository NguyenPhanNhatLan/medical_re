import torch
import torch.nn as nn

class BERTSE(nn.Module):
    def __init__(self, encoder, hidden_size: int, num_labels: int):
        super().__init__()
        self.encoder = encoder  # PhoBERT hoặc ViHealthBERT
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        """
        e1_pos, e2_pos: List[(start_idx, end_idx)] theo token index
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state  # [B, L, H]

        # 1. CLS vector
        cls_vec = hidden[:, 0, :]  # [B, H]

        # 2. Entity Start vectors
        e1_vecs = []
        e2_vecs = []

        for i in range(input_ids.size(0)):
            s1, _ = e1_pos[i]
            s2, _ = e2_pos[i]

            # +1 vì token sau <e1> / <e2>
            e1_vecs.append(hidden[i, s1 + 1])
            e2_vecs.append(hidden[i, s2 + 1])

        e1_vecs = torch.stack(e1_vecs)  # [B, H]
        e2_vecs = torch.stack(e2_vecs)  # [B, H]

        # 3. Concatenate
        x = torch.cat([cls_vec, e1_vecs, e2_vecs], dim=-1)  # [B, 3H]

        logits = self.classifier(x)
        return logits
