import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import math

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer_path, block_size=128):
        self.block_size = block_size
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
 
 
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        
        self.sep_id = self.tokenizer.token_to_id("[SEP]")

        if not isinstance(texts, list):
             text_list = texts['text']
        else:
             text_list = texts
        
        valid_texts = [t for t in text_list if len(t.strip()) > 0]
        
        encodings = self.tokenizer.encode_batch(valid_texts)
        full_token_ids = []
        for enc in encodings:
            full_token_ids.extend(enc.ids + [self.sep_id])
            
        self.tokens = torch.tensor(full_token_ids, dtype=torch.long)
        total_tokens = self.tokens.size(0)
        
        self.num_samples = math.ceil(total_tokens / block_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = min(start_idx + self.block_size, len(self.tokens))
        
        chunk = self.tokens[start_idx:end_idx]
        
        if len(chunk) < self.block_size:
            pad_len = self.block_size - len(chunk)
            pads = torch.full((pad_len,), self.pad_id, dtype=torch.long)
            input_ids = torch.cat([chunk, pads])
            
            labels = input_ids.clone()
            labels[-pad_len:] = -100 
        else:
            input_ids = chunk
            labels = input_ids.clone()
            
        return {
            "input_ids": input_ids,
            "labels": labels
        }
