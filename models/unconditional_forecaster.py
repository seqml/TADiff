import torch
import torch.nn as nn

from models.diffusion.tadiff import TADiff

from samplers import DDPMSampler, DDIMSampler
import numpy as np
import time
import random


class UnConditionalForecaster(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs["device"]
        self.configs = configs
        self._init_diff(configs["diffusion"])

    def _init_diff(self, configs):
        input_dim = 1
        configs["device"] = self.device
        if configs["type"] == "TADiff":
            self.diff_model = TADiff(configs, input_dim).to(self.device)
        
        self.num_steps = configs["num_steps"]

        self.ddpm = DDPMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)
        self.ddim = DDIMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)
    
    def _noise_estimation_loss(self, x, mask, tp, attr_emb, t):
        noise = torch.randn_like(x)
        noisy_x = self.ddpm.forward(x, t, noise)
        noisy_x = noisy_x * mask + x * (1-mask)
        ret_dict = self.predict_noise(noisy_x, mask, tp, attr_emb, t)
        pred_noise = ret_dict["pred_noise"]
        loss_dict = ret_dict["loss_dict"]
        residual = (noise - pred_noise) * mask
        loss_dict["noise_loss"] = (residual ** 2).mean()
        all_loss = torch.zeros_like(loss_dict["noise_loss"])
        for k in loss_dict.keys():
            all_loss += loss_dict[k]
        loss_dict["all"] = all_loss
        return loss_dict, ret_dict

    def predict_noise(self, xt, mask, tp, attr_emb, t):
        noisy_x = torch.unsqueeze(xt, 1)
        ret_dict = self.diff_model(noisy_x, mask, tp, attr_emb, t)
        return ret_dict