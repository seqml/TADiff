import torch
import torch.nn as nn
import torch.nn.functional as F

class TextProjectorAvg(nn.Module):
    def __init__(self, n_var=1, dim_in=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_var = n_var
        self.proj_out = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, attr):
        B = attr.shape[0]
        h = torch.mean(attr, dim=1, keepdim=True)
        h = h[:,None,:,:].expand([-1, self.n_var, -1, -1])
        h = h.reshape((B, self.n_var, 1, -1))
        out = self.proj_out(h)
        return out