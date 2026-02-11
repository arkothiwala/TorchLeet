import torch
import numpy as np
class SinusoidalPositionalEmbeddings(torch.nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim

    def get_sine_wave(self, idx):
        return np.sin(idx/(10000**(np.arange(self.n_dim)/self.n_dim)))

    def get_cos_wave(self, idx):
        return np.cos(idx/(10000**(np.arange(self.n_dim)/self.n_dim)))

    def forward(self, idx):
        if isinstance(idx, int):
            print(idx)
            if idx%2 == 0:
                return self.get_sine_wave(idx)
            elif idx%2 == 1:
                return self.get_cos_wave(idx)
        else:
            sine_op = self.get_sine_wave(idx)*(1-np.arange(self.n_dim)%2)
            cos_op = self.get_cos_wave(idx)*(np.arange(self.n_dim)%2)
            return sine_op+cos_op