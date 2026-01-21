# utils/collator.py
import torch
from torch.nn.utils.rnn import pad_sequence

class RBERTCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        e1_masks = [item['e1_mask'] for item in batch]
        e2_masks = [item['e2_mask'] for item in batch]
        labels = torch.tensor([item['label_id'] for item in batch], dtype=torch.long)

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        e1_masks_padded = pad_sequence(e1_masks, batch_first=True, padding_value=0)
        e2_masks_padded = pad_sequence(e2_masks, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "e1_mask": e1_masks_padded,
            "e2_mask": e2_masks_padded,
            "labels": labels
        }
        
class BERTESCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        # Pos là tensor 1 chiều hoặc scalar, stack lại thành [Batch_size]
        e1_pos = torch.stack([item['e1_pos'] for item in batch])
        e2_pos = torch.stack([item['e2_pos'] for item in batch])
        
        labels = torch.tensor([item['label_id'] for item in batch], dtype=torch.long)

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "e1_pos": e1_pos,
            "e2_pos": e2_pos,
            "labels": labels
        }
