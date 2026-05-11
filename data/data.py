import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
import time

class CounterfactualForecastDataset:
    def __init__(self, folder, text_type, hist_len, pred_len, **kwargs):
        super().__init__()
        self.folder = folder
        self.text_type = text_type
        self.hist_len = hist_len
        self.pred_len = pred_len

    def get_split(self, split, *args):
        return CounterfactualForecastSplit(self.folder, split, self.text_type, self.hist_len, self.pred_len)

class CounterfactualForecastSplit(Dataset):
    def __init__(self, folder, split, text_type, hist_len, pred_len):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split
        self.folder = folder
        self.text_type = text_type
        self.hist_len = hist_len
        self.pred_len = pred_len
        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        
        if "CF" in self.text_type:
            self.ts = np.load(os.path.join(self.folder, f"{self.text_type}/{self.split}_ts.npy"))
            self.caps = np.load(fr"{self.folder}/{self.text_type}/{self.split}_caps.npy", allow_pickle=True)
        else:
            self.ts = np.load(os.path.join(self.folder, f"{self.split}_ts.npy"))
            self.caps = np.load(fr"{self.folder}/{self.split}_caps.npy", allow_pickle=True)

        self.n_samples = len(self.ts)
        self.n_steps = self.hist_len + self.pred_len
        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        tmp_ts = self.ts[idx]
        tmp_cap = self.caps[idx]
        if len(tmp_ts.shape) == 1:
            tmp_ts = tmp_ts[...,np.newaxis]
        if isinstance(tmp_cap, np.ndarray):
            tmp_cap = tmp_cap[0]
        tmp_mask = np.concatenate([np.zeros(self.hist_len), np.ones(self.pred_len)], axis=0)
        hist_mean = np.mean(tmp_ts[:self.hist_len])
        hist_std = np.std(tmp_ts[:self.hist_len])   
        if hist_std < 1e-5:
            hist_mean = 0
            hist_std = 1
        hist_mean = np.array(hist_mean, dtype=np.float64)[None,None]
        hist_std = np.array(hist_std, dtype=np.float64)[None,None]

        data_dict = {"ts": tmp_ts,
                "mask": tmp_mask,
                "hist_len": self.hist_len,
                "pred_len": self.pred_len,
                "tp": self.time_point,
                "hist_mean": hist_mean,
                "hist_std": hist_std}

        if self.text_type != "none":
            data_dict["hist_cap"] = self.caps[idx][0]
            data_dict["pred_cap"] = self.caps[idx][1]

        return data_dict

    def __len__(self):
        return self.n_samples