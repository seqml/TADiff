import os
import time
import torch
import numpy as np
from .dttc import DTTC
import yaml
import tqdm
import numpy as np
from scipy import linalg
import random
import torch.nn.functional as F

class BaseEvaluator:
    def __init__(self, configs, dataset, model):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_data(dataset)
        if "dttc_config_path" in configs.keys():
            self._init_dttc(configs)

    def _init_dttc(self, configs):
        dttc_configs = yaml.safe_load(open(configs["dttc_config_path"]))
        self.dttc = DTTC(dttc_configs)
        self.dttc.load_state_dict(torch.load(configs["dttc_model_path"]))
        self.dttc = self.dttc.to(self.dttc.device)

    def _init_cfgs(self, configs):
        self.configs = configs
        self.batch_size = self.configs["batch_size"]
        self.n_samples = self.configs["n_samples"]
        self.model_path = self.configs["model_path"]

    def _init_model(self, model):
        self.model = model
        if self.model_path != "":
            print("Loading pretrained model from {}".format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path))

    def _init_data(self, dataset):
        self.dataset = dataset
        self.valid_loader = dataset.get_loader(split="valid", batch_size=self.batch_size, shuffle=False, include_self=False)
        self.test_loader = dataset.get_loader(split="test", batch_size=self.batch_size, shuffle=False, include_self=False)

    def evaluate(self, split, sampler="ddpm", is_determin=True):

        self.model.eval()
        sample_num, mae, mse, dttc_i, dttc_e = 0, 0, 0, 0, 0
        with torch.no_grad():
            if split == "valid":
                tmp_loader = self.valid_loader
            elif split == "test":
                tmp_loader = self.test_loader
            for batch_no, batch in enumerate(tmp_loader):
                multi_preds = self.model.forecast(batch, self.n_samples, sampler, is_determin)
                multi_preds = multi_preds.permute(0,1,3,2)
                pred = multi_preds.median(dim=0).values
                ts = batch["ts"].to(self.model.device).float()
                hist_len = batch["hist_len"][0]
                hist_gt_ts = ts[:, :hist_len]
                pred_gt_ts = ts[:, hist_len:]
                pred_gen_ts = pred[:, hist_len:]
                pred_cap = batch["pred_cap"]

                dttc_i += self.dttc.hist_pred_sim(hist_gt_ts, pred_gen_ts)
                dttc_e += self.dttc.pred_text_sim(pred_gen_ts, pred_cap)
                mse += torch.nn.functional.mse_loss(pred_gen_ts, pred_gt_ts).item() * pred.shape[0]
                mae += torch.nn.functional.l1_loss(pred_gen_ts, pred_gt_ts).item() * pred.shape[0]
                sample_num += pred.shape[0]

        if sample_num > 0:
            mae /= sample_num
            mse /= sample_num
            dttc_i /= sample_num
            dttc_e /= sample_num

        res_dict = {
            "tensorboard":{},
            "df":{},
        }
        if sample_num > 0:
            res_dict["tensorboard"].update({"mse":mse, "mae":mae, "dttc_i":dttc_i, "dttc_e":dttc_e})
            res_dict["df"].update({"mse":mse, "mae":mae, "dttc_i":dttc_i, "dttc_e":dttc_e})

            print("MSE: ", mse)
            print("MAE: ", mae)
            print("DTTC-I: ", dttc_i)
            print("DTTC-E: ", dttc_e)
        return res_dict
