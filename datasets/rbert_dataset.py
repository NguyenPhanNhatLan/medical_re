# datasets/rbert_dataset.py
import torch
from datasets.base_dataset import BaseREDataset, SPECIAL_TOKENS
class RBERTDataset(BaseREDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_end_tags = True  # R-BERT bắt buộc cần thẻ đóng

        # 1. Kiểm tra và thêm special tokens nếu chưa có
        if self.tokenizer.unk_token_id is not None:
             e1_check_id = self.tokenizer.convert_tokens_to_ids("<e1>")
             if e1_check_id == self.tokenizer.unk_token_id:
                 print(f"[RBERTDataset] Adding special tokens {SPECIAL_TOKENS} to tokenizer...")
                 self.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
                 # Lưu ý quan trọng: Nhớ resize embedding của model trong file train.py!

        self.e1_start_id = self.tokenizer.convert_tokens_to_ids("<e1>")
        self.e1_end_id   = self.tokenizer.convert_tokens_to_ids("</e1>")
        self.e2_start_id = self.tokenizer.convert_tokens_to_ids("<e2>")
        self.e2_end_id   = self.tokenizer.convert_tokens_to_ids("</e2>")

    def _find_first(self, ids: torch.Tensor, token_id: int) -> int:
        matches = (ids == token_id).nonzero(as_tuple=True)[0]
        return matches[0].item() if len(matches) > 0 else 0

    def build_span_masks(self, input_ids: torch.Tensor):
        # Sử dụng trực tiếp IDs đã cache ở __init__, không cần tạo dict mới
        e1s = self._find_first(input_ids, self.e1_start_id)
        e1e = self._find_first(input_ids, self.e1_end_id)
        e2s = self._find_first(input_ids, self.e2_start_id)
        e2e = self._find_first(input_ids, self.e2_end_id)
            
        L = input_ids.size(0)
        e1_mask = torch.zeros(L, dtype=torch.long)
        e2_mask = torch.zeros(L, dtype=torch.long)
        
        # Tạo mask cho các token NẰM GIỮA thẻ mở và thẻ đóng
        # [e1s+1 : e1e] là python slice, nó lấy từ sau thẻ <e1> đến trước thẻ </e1>
        if e1e > e1s + 1:
            e1_mask[e1s + 1 : e1e] = 1
        if e2e > e2s + 1:
            e2_mask[e2s + 1 : e2e] = 1

        return e1_mask, e2_mask

    def __getitem__(self, idx):
        out = self.get_common_fields(idx)
   
        if not isinstance(out["input_ids"], torch.Tensor):
             out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)

        # Tạo mask
        e1_mask, e2_mask = self.build_span_masks(out["input_ids"])
        
        out["e1_mask"] = e1_mask
        out["e2_mask"] = e2_mask
        
        # Thêm length để tiện debug hoặc dùng cho pack_padded_sequence nếu cần
        out["length"] = out["input_ids"].size(0)
        
        return out
