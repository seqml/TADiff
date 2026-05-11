import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class TextEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.device = configs["device"]
        self.emb_dim = configs["text_emb"]
        self.vocab_emb = nn.Embedding(num_embeddings=configs["word_size"], embedding_dim=self.emb_dim)
        self.trans_layer = get_torch_trans(heads=8, layers=2, channels=self.emb_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(configs["tokenizer_path"])
        self.init_pe(self.emb_dim)
    
    def init_pe(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False
    
    def forward(self, text, task_type):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")["input_ids"]
        if task_type == "F":
            task_token = self.tokenizer(["<forecasting>"], return_tensors="pt")["input_ids"]
        elif task_type == "A":
            task_token = self.tokenizer(["<attribution>"], return_tensors="pt")["input_ids"]
        task_token = task_token.expand(inputs.shape[0], task_token.shape[1])
        inputs = torch.cat([task_token, inputs[:,1:]], dim=1)

        inputs = inputs.to(self.device)
        text_emb = self.vocab_emb(inputs)
        text_emb += self.pe[:, :text_emb.shape[1], :].to(text_emb.device)
        text_emb = self.trans_layer(text_emb)
        return text_emb