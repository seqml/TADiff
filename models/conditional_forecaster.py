import torch
import torch.nn as nn
import numpy as np
import random
from models.encoders.text_encoder import TextEncoder
from models.unconditional_forecaster import UnConditionalForecaster
from models.encoders.cond_projector import TextProjectorAvg

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class ConditionalForecaster(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"]
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self._init_condition_encoders(cond_configs)
        self._init_diff(diff_configs)
        self.dttc_i_weight = cond_configs["dttc_loss"]["dttc_i_weight"]
        self.dttc_e_weight = cond_configs["dttc_loss"]["dttc_e_weight"]

    def choose_textencoder(self, cond_modal, cond_configs):
        cond_configs["device"] = self.device
        if cond_modal == "text":
            text_en = TextEncoder(cond_configs).to(self.device)
        if self.cond_configs["text_projector"] == "uniform":
            cond_projector = TextProjectorAvg(dim_in=self.cond_configs["text_emb"],
                                              dim_out=self.diff_configs["diffusion"]["channels"])
        cond_projector = cond_projector.to(self.device)
        return text_en, cond_projector

    def _init_condition_encoders(self, cond_configs):
        if cond_configs["cond_modal"] != "none":
            self.text_en, self.cond_projector = self.choose_textencoder(cond_configs["cond_modal"], cond_configs[cond_configs["cond_modal"]])

    def _init_diff(self, configs):
        configs["device"] = self.device
        configs["diffusion"]["text_projector"] = self.cond_configs["text_projector"]
        self.forecaster = UnConditionalForecaster(configs=configs)
        if configs["forecaster_pretrain_path"] != "":
            self.load_state_dict(torch.load(configs["forecaster_pretrain_path"]))
            print("Load the pretrain forecaster")
        else:
            print("Learn from scratch")

    def forward(self, batch, task_type, loss_type, dttc_model=None, is_train=True):
        x, mask, tp, text, mean, std = self._unpack_data(batch, task_type)

        if self.cond_configs["cond_modal"] != "none":
            text_emb_raw = self.text_en(text, task_type)
            text_emb = self.cond_projector(text_emb_raw)
        else:
            text_emb = None

        B = x.shape[0]
        if loss_type == "noise":
            if is_train:
                t = torch.randint(0, self.forecaster.num_steps, [B], device=self.device)
                loss, _ = self.forecaster._noise_estimation_loss(x, mask, tp, text_emb, t)
                return loss
            loss_dict = {}
            for t in range(self.forecaster.num_steps):
                t = (torch.ones(B, device=self.device) * t).long()
                tmp_loss_dict, _ = self.forecaster._noise_estimation_loss(x, mask, tp, text_emb, t)
                for k in tmp_loss_dict:
                    if k in loss_dict.keys():
                        loss_dict[k] += tmp_loss_dict[k]
                    else:
                        loss_dict[k] = tmp_loss_dict[k]
            for k in loss_dict:
                loss_dict[k] = loss_dict[k] / self.forecaster.num_steps
            return loss_dict
        
        elif loss_type == "dttc":
            xt = torch.randn_like(x)
            xt = xt * mask + x * (1-mask)
            loss_dict = {}
            target_t = random.randint(0, self.forecaster.num_steps-1) - 1

            for t in range(self.forecaster.num_steps-1, target_t, -1):
                self.zero_grad()
                dttc_model.zero_grad()
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                ret_dict = self.forecaster.predict_noise(xt, mask, tp, text_emb, t)
                pred_x0 = self.forecaster.ddim.predict_x0(xt, ret_dict["pred_noise"], t)
                hist_len = batch["hist_len"][0]

                pred_x0 = pred_x0 * std + mean
                pred_x0 = pred_x0.permute(0,2,1)
                gt_x = x.permute(0,2,1)
                loss_dict["dttc_i"] = dttc_model.hist_pred_sim(gt_x[:, :hist_len], pred_x0[:, hist_len:], with_grad=True) / B
                loss_dict["dttc_e"] = dttc_model.pred_text_sim(pred_x0[:, hist_len:], text, with_grad=True) / B
                loss_dict["all"] = -1.0 * (loss_dict["dttc_i"] * self.dttc_i_weight + loss_dict["dttc_e"] * self.dttc_e_weight)

                xt = self.forecaster.ddim.reverse(xt, ret_dict["pred_noise"], t, noise, is_determin=True)
                xt = xt * mask + x * (1-mask)
                
            return loss_dict

    def _unpack_data(self, batch, task_type):
        ts = batch["ts"].to(self.device).float()
        tp = batch["tp"].to(self.device).float()
        mask = batch["mask"].to(self.device).float()
        if task_type == "A":
            hist_len = batch["hist_len"][0]
            ts = ts[:, :hist_len]
            mask = 1 - mask[:, :hist_len]
            tp = tp[:, :hist_len]
            text = batch["hist_cap"]
            mean = torch.zeros_like(batch["hist_mean"]).to(self.device).float()
            std = torch.ones_like(batch["hist_std"]).to(self.device).float()
        elif task_type == "F":
            text = batch["pred_cap"]
            mean = torch.zeros_like(batch["hist_mean"]).to(self.device).float()
            std = torch.ones_like(batch["hist_std"]).to(self.device).float()
        ts = ts.permute(0, 2, 1)
        ts = (ts - mean) / std
        mask = mask[:,None,:]
        return ts, mask, tp, text, mean, std

    def forecast(self, batch, n_samples, sampler="ddim", is_determin=True, return_initial_noise=False):
        return self.forecast_CF(batch, n_samples, sampler, is_determin=is_determin, return_initial_noise=return_initial_noise)
    
    
    @torch.no_grad()
    def forecast_CF(self, batch, n_samples, sampler="ddim", is_determin=True, return_initial_noise=False):
        hist_ts, hist_mask, hist_tp, hist_text, _, _ = self._unpack_data(batch, "A")
        if self.cond_configs["cond_modal"] != "none":
            hist_text_emb_raw = self.text_en(hist_text, "A")
            hist_text_emb = self.cond_projector(hist_text_emb_raw)
        else:
            hist_text_emb = None

        ts, mask, tp, pred_text, mean, std = self._unpack_data(batch, "F")
        if self.cond_configs["cond_modal"] != "none":
            pred_text_emb_raw = self.text_en(pred_text, "F")
            pred_text_emb = self.cond_projector(pred_text_emb_raw)
        else:
            pred_text_emb = None

        samples = []
        initial_noise = []
        B = ts.shape[0]
        for i in range(n_samples):
            hist_xt = hist_ts
            for t in range(self.forecaster.num_steps-1):
                t = (torch.ones(B, device=self.device) * t).long()
                ret_dict = self.forecaster.predict_noise(hist_xt, hist_mask, hist_tp, hist_text_emb, t)
                hist_xt = self.forecaster.ddim.forward(hist_xt, ret_dict["pred_noise"], t)
            hist_len = hist_xt.shape[2]
            x = torch.randn_like(ts)
            x[:, :, :hist_len] = ts[:, :, :hist_len]
            x[:, :, hist_len:] = hist_xt
            initial_noise.append(x)
            for t in range(self.forecaster.num_steps-1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                ret_dict = self.forecaster.predict_noise(x, mask, tp, pred_text_emb, t)
                if sampler == "ddpm":
                    x = self.forecaster.ddpm.reverse(x, ret_dict["pred_noise"], t, noise)
                else:
                    x = self.forecaster.ddim.reverse(x, ret_dict["pred_noise"], t, noise, is_determin=is_determin)
                x = x * mask + ts * (1-mask)

            x = x * std + mean
            samples.append(x)
        
        if return_initial_noise:
            return torch.stack(samples), torch.stack(initial_noise)
        else:
            return torch.stack(samples)