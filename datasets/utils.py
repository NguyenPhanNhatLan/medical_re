import torch

def find_entity_positions(input_ids: torch.Tensor, e1_id: int, e2_id: int):
    """
    Tìm vị trí bắt đầu của <e1> và <e2> trong input_ids.
    Hoạt động với cả 1D Tensor [Length] (trong __getitem__) 
    hoặc 2D Tensor [Batch, Length] (trong forward).
    
    Args:
        input_ids (torch.Tensor): Tensor chứa token IDs.
        e1_id (int): ID của token <e1>.
        e2_id (int): ID của token <e2>.
        
    Returns:
        e1_pos, e2_pos: Tensor chứa vị trí index.
    """
    # 1. Tạo mask boolean: True tại nơi có token, False tại nơi khác
    # logic gốc từ vihealth_encoder.py
    e1_mask = (input_ids == e1_id)
    e2_mask = (input_ids == e2_id)

    # 2. Dùng argmax để lấy vị trí True đầu tiên
    # Chuyển sang int() trước khi argmax để đảm bảo tính tương thích
    # dim=-1 giúp code chạy đúng cho cả input 1D (vector) và 2D (batch)
    e1_pos = torch.argmax(e1_mask.int(), dim=-1)
    e2_pos = torch.argmax(e2_mask.int(), dim=-1)

    return e1_pos, e2_pos