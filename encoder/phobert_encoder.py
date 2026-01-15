import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Any, Optional

class PhoBERTEncoder:
    def __init__(
        self, 
        model_name: str = "vinai/phobert-base", 
        max_length: int = 256, 
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set to evaluation mode for feature extraction

        # Define and inject special entity markers
        self.special_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"]
        num_added_toks = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.special_tokens}
        )
        
        if num_added_toks > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Cache special token IDs for efficient lookup during encoding
        self.e1_start_id = self.tokenizer.convert_tokens_to_ids("<e1>")
        self.e1_end_id = self.tokenizer.convert_tokens_to_ids("</e1>")
        self.e2_start_id = self.tokenizer.convert_tokens_to_ids("<e2>")
        self.e2_end_id = self.tokenizer.convert_tokens_to_ids("</e2>")

    def _inject_markers(
        self, 
        text: str, 
        e1: Dict[str, Any], 
        e2: Dict[str, Any]
    ) -> str:
        # Determine order of entities to handle offset shifts correctly
        if e1["start"] < e2["start"]:
            first, second = e1, e2
            first_tags = ("<e1>", "</e1>")
            second_tags = ("<e2>", "</e2>")
        else:
            first, second = e2, e1
            first_tags = ("<e2>", "</e2>")
            second_tags = ("<e1>", "</e1>")

        # Inject markers for the first entity
        text = (
            text[:first["start"]] + 
            first_tags[0] + 
            text[first["start"]:first["end"]] + 
            first_tags[1] + 
            text[first["end"]:]
        )

        # Calculate offset shift caused by the first pair of tags
        offset = len(first_tags[0]) + len(first_tags[1])

        # Inject markers for the second entity
        text = (
            text[:second["start"] + offset] + 
            second_tags[0] + 
            text[second["start"] + offset:second["end"] + offset] + 
            second_tags[1] + 
            text[second["end"] + offset:]
        )

        return text

    def encode(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        # 1. Text Preprocessing & Tokenization
        marked_text = self._inject_markers(
            sample["text"], 
            sample["entity_1"], 
            sample["entity_2"]
        )
        
        encoding = self.tokenizer(
            marked_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 2. Forward Pass (Inference Mode)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state  # Shape: [1, Seq_Len, Hidden_Size]
        cls_vector = last_hidden_state[0, 0]           # [CLS] token vector

        # 3. Entity Vector Extraction
        token_ids = input_ids[0]
        
        try:
            e1_s = (token_ids == self.e1_start_id).nonzero()[0].item()
            e1_e = (token_ids == self.e1_end_id).nonzero()[0].item()
            e2_s = (token_ids == self.e2_start_id).nonzero()[0].item()
            e2_e = (token_ids == self.e2_end_id).nonzero()[0].item()

            # --- Strategy A: BERT-ES (Entity Start) ---
            # Uses the vector of the first token inside the entity tags
            es_e1 = last_hidden_state[0, e1_s + 1]
            es_e2 = last_hidden_state[0, e2_s + 1]

            # --- Strategy B: R-BERT (Mean Pooling) ---
            # Computes the mean of all vectors within the entity span (exclusive of tags)
            # Span range: [start_index + 1 : end_index]
            rbert_e1 = last_hidden_state[0, e1_s + 1 : e1_e].mean(dim=0)
            rbert_e2 = last_hidden_state[0, e2_s + 1 : e2_e].mean(dim=0)

        except (IndexError, RuntimeError) as e:
            hidden_size = self.model.config.hidden_size
            fallback_vec = torch.zeros(hidden_size).to(self.device)
            
            es_e1, es_e2 = fallback_vec, fallback_vec
            rbert_e1, rbert_e2 = fallback_vec, fallback_vec        
            print(f"Warning: Entity extraction failed for text: {sample['text'][:30]}...")

        return {
            "cls": cls_vector.cpu(),
            "es_e1": es_e1.cpu(),
            "es_e2": es_e2.cpu(),
            "rbert_e1": rbert_e1.cpu(),
            "rbert_e2": rbert_e2.cpu(),
            "relation": sample.get("relation")
        }

# if __name__ == "__main__":
 
#     encoder = PhoBERTEncoder(device="cpu")
#     sample_data = {
#         "text": "Bệnh nhân được chẩn đoán viêm phổi và điều trị bằng kháng sinh.",
#         "entity_1": {"start": 27, "end": 36}, # viêm phổi
#         "entity_2": {"start": 55, "end": 64}, # kháng sinh
#         "relation": "treats"
#     }
#     features = encoder.encode(sample_data)

#     print("Encoding successful!")
#     print(f"CLS Shape      : {features['cls'].shape}")
#     print(f"BERT-ES E1     : {features['es_e1'].shape}")
#     print(f"R-BERT E1 (Avg): {features['rbert_e1'].shape}")