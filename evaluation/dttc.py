import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import numpy as np

from .patchtst_modules import PatchEmbedding, Encoder, EncoderLayer, AttentionLayer, FullAttention
import math

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class TSEncoder(nn.Module):
    def __init__(self, configs):
        super(TSEncoder, self).__init__()
        self.patch_encoder = PatchEncoder(configs=configs)
        if configs["pretrain_encoder_path"] != "":
            pretrain_encoder_path = configs["pretrain_encoder_path"]
            print(f"Load pretrain ts encoder from {pretrain_encoder_path}")
            self.patch_encoder.load_state_dict(torch.load(configs["pretrain_encoder_path"]))

        self.time_transformer = get_torch_trans(heads=configs["n_heads"], layers=1, channels=configs["d_model"])
        self.var_transformer = get_torch_trans(heads=configs["n_heads"], layers=1, channels=configs["d_model"])
        self.out_projector = nn.Linear(configs["d_model"], configs["coemb_dim"])
    
    def forward(self, ts):
        B, L, N = ts.shape
        ts_var_emb = self.patch_encoder(ts)
        var_emb = self.time_transformer(ts_var_emb)[:,:1,:].reshape(B, N, -1)
        co_emb = self.var_transformer(var_emb)[:,:1,:].reshape(B, -1)
        co_emb = self.out_projector(co_emb)
        return co_emb

class PatchEncoder(nn.Module):
    def __init__(self, configs):
        super(PatchEncoder, self).__init__()
        self.device = configs["device"]
        self.patch_embedding = PatchEmbedding(configs["d_model"], configs["patch_len"], 1, configs["stride"], configs["padding"], configs["dropout"])
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs["factor"], attention_dropout=configs["dropout"],
                                      output_attention=configs["output_attention"]), configs["d_model"], configs["n_heads"]),
                    configs["d_model"],
                    configs["d_ff"],
                    dropout=configs["dropout"],
                    activation=configs["activation"]
                ) for l in range(configs["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(configs["d_model"])
        )
        patch_seq_len = int((configs["seq_len"] - configs["patch_len"]) / configs["stride"] + 2)
        tp = torch.tensor([i for i in range(patch_seq_len)]).to(self.device)
        tp = tp[None]
        self.time_pos_emb = self.time_embedding(tp, d_model=configs["d_model"])
        self.time_pos_emb.requires_grad = False
        self.var_pos_emb = nn.Embedding(num_embeddings=configs["n_var"], embedding_dim=configs["d_model"]).to(self.device)
    
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, ts):
        B, L, N = ts.shape
        ts = ts.permute(0, 2, 1).reshape(B*N, 1, L)
        ts_emb = self.patch_embedding(ts)
        BN, Nl, D = ts_emb.shape
        timposemb = self.time_pos_emb.expand((BN,-1,-1))
        varposemb = self.var_pos_emb(torch.arange(N).to(self.device))[None].expand(B,-1,-1)
        varposemb = varposemb.reshape(B*N, 1, -1)
        ts_emb += timposemb + varposemb
        ts_enc_out, attns = self.encoder(ts_emb)
        return ts_enc_out
    
class TextEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.device = configs["device"]
        self.emb_dim = configs["text_emb"]
        self.vocab_emb = nn.Embedding(num_embeddings=configs["word_size"], embedding_dim=self.emb_dim)
        self.trans_layer = get_torch_trans(heads=8, layers=2, channels=self.emb_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(configs["tokenizer_path"])
        self.init_pe(self.emb_dim)

        self.text_enc = nn.Sequential(
            nn.Linear(configs["text_emb"], configs["textemb_hidden_dim"]),
            nn.LayerNorm(configs["textemb_hidden_dim"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(configs["textemb_hidden_dim"], configs["coemb_dim"])
        )
    
    def init_pe(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False
    
    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")["input_ids"]
        inputs = inputs.to(self.device)
        text_emb = self.vocab_emb(inputs)
        text_emb += self.pe[:, :text_emb.shape[1], :].to(text_emb.device)
        text_emb = torch.mean(self.trans_layer(text_emb), dim=1)
        text_emb = self.text_enc(text_emb)
        return text_emb

class DTTC(nn.Module):
    def __init__(self, configs):
        super(DTTC, self).__init__()
        configs["text"]["device"] = configs["device"]
        configs["ts"]["device"] = configs["device"]

        if configs["text"]["type"] == "vanilla":
            self.text_enc = TextEncoder(configs["text"])
        if configs["ts"]["type"] == "patch_encoder":
            self.ts_enc = TSEncoder(configs["ts"])

        self.device = configs["device"]
        self.configs = configs
        self.ContrastiveLoss = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, hist_ts, pred_ts, hist_text, pred_text):
        B = hist_ts.shape[0]
        hist_emb = self.ts_enc(hist_ts)
        hist_pres_emb = hist_emb[:,:self.configs["ts"]["coemb_dim"]//2]
        hist_edit_emb = hist_emb[:,self.configs["ts"]["coemb_dim"]//2:]
        hist_text_emb = self.text_enc(hist_text)
        pred_emb = self.ts_enc(pred_ts)
        pred_pres_emb = pred_emb[:,:self.configs["ts"]["coemb_dim"]//2]
        pred_edit_emb = pred_emb[:,self.configs["ts"]["coemb_dim"]//2:]
        pred_text_emb = self.text_enc(pred_text)
        loss_dict = {}

        hist_pred_sim = torch.mm(hist_pres_emb, pred_pres_emb.permute(1,0))
        labels = torch.arange(hist_pred_sim.shape[0], device=hist_pred_sim.device)
        hist_pred_loss0 = self.ContrastiveLoss(hist_pred_sim, labels)
        hist_pred_sim = hist_pred_sim.permute(1,0)
        hist_pred_loss1 = self.ContrastiveLoss(hist_pred_sim, labels)
        hist_pred_loss0 = torch.mean(hist_pred_loss0, dim=-1)
        hist_pred_loss1 = torch.mean(hist_pred_loss1, dim=-1)
        loss_dict["hist_pred"] = (hist_pred_loss0 + hist_pred_loss1) / 2

        hist_text_sim = torch.mm(hist_edit_emb, hist_text_emb.permute(1,0))
        labels = torch.arange(hist_text_sim.shape[0], device=hist_text_sim.device)
        hist_text_loss0 = self.ContrastiveLoss(hist_text_sim, labels)
        hist_text_sim = hist_text_sim.permute(1,0)
        hist_text_loss1 = self.ContrastiveLoss(hist_text_sim, labels)
        hist_text_loss0 = torch.mean(hist_text_loss0, dim=-1)
        hist_text_loss1 = torch.mean(hist_text_loss1, dim=-1)
        loss_dict["hist_text"] = (hist_text_loss0 + hist_text_loss1) / 2

        pred_text_sim = torch.mm(pred_edit_emb, pred_text_emb.permute(1,0))
        labels = torch.arange(pred_text_sim.shape[0], device=pred_text_sim.device)
        pred_text_loss0 = self.ContrastiveLoss(pred_text_sim, labels)
        pred_text_sim = pred_text_sim.permute(1,0)
        pred_text_loss1 = self.ContrastiveLoss(pred_text_sim, labels)
        pred_text_loss0 = torch.mean(pred_text_loss0, dim=-1)
        pred_text_loss1 = torch.mean(pred_text_loss1, dim=-1)
        loss_dict["pred_text"] = (pred_text_loss0 + pred_text_loss1) / 2

        loss_dict["all"] = loss_dict["hist_pred"] * self.configs["hist_pred_weight"] + (loss_dict["hist_text"] + loss_dict["pred_text"])  * self.configs["pred_text_weight"] / 2

        return loss_dict
    
    def hist_pred_sim(self, hist_ts, pred_ts, with_grad=False):
        hist_pres_emb = self.ts_enc(hist_ts)[:,:self.configs["ts"]["coemb_dim"]//2]
        pred_pres_emb = self.ts_enc(pred_ts)[:,:self.configs["ts"]["coemb_dim"]//2]
        if with_grad is False:
            sim = torch.mm(hist_pres_emb, pred_pres_emb.permute(1,0)).trace().item()
        else:
            sim = torch.mm(hist_pres_emb, pred_pres_emb.permute(1,0)).trace()
        return sim

    def pred_text_sim(self, pred_ts, text, with_grad=False):
        pred_edit_emb = self.ts_enc(pred_ts)[:,self.configs["ts"]["coemb_dim"]//2:]
        text_emb = self.text_enc(text)
        if with_grad is False:
            sim = torch.mm(pred_edit_emb, text_emb.permute(1,0)).trace().item()
        else:
            sim = torch.mm(pred_edit_emb, text_emb.permute(1,0)).trace()
        return sim

    def get_ts_coemb(self, ts):
        ts_co_emb = self.ts_enc(ts)
        return ts_co_emb
    
    def get_text_coemb(self, text):
        text_co_emb = self.text_enc(text)
        return text_co_emb