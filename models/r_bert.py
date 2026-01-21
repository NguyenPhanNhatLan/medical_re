# models/r_bert.py
import torch
import torch.nn as nn

class RBERT(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels, dropout_rate):
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        
        self.cls_fc = nn.Linear(hidden_size, hidden_size)
        self.entity_fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh() 
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask, labels=None):
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 1. Lấy vector
        cls_vec = enc_out.cls # [B, H]
        hidden  = enc_out.hidden_states # [B, L, H]

        # 2. Pooling entities
        e1_vec = self.encoder.masked_mean_pool(hidden, e1_mask) # [B, H]
        e2_vec = self.encoder.masked_mean_pool(hidden, e2_mask) # [B, H]

        # 3. Apply Activation (Refinement) - Quan trọng cho R-BERT
        cls_vec = self.activation(self.cls_fc(cls_vec))
        e1_vec = self.activation(self.entity_fc(e1_vec))
        e2_vec = self.activation(self.entity_fc(e2_vec))

        # 4. Dropout & Concat
        cls_vec = self.dropout(cls_vec)
        e1_vec = self.dropout(e1_vec)
        e2_vec = self.dropout(e2_vec)

        features = torch.cat([cls_vec, e1_vec, e2_vec], dim=-1) # [B, 3H]
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
