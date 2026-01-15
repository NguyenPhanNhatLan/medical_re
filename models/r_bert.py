import torch
import torch.nn as nn

class RBERT(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask):
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_vec = enc_out.cls
        hidden  = enc_out.hidden_states

        e1_vec = self.encoder.masked_mean_pool(hidden, e1_mask)
        e2_vec = self.encoder.masked_mean_pool(hidden, e2_mask)

        x = torch.cat([cls_vec, e1_vec, e2_vec], dim=-1)
        return self.classifier(x)
