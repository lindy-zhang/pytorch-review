import torch
from torch import Tensor
import torch.nn as nn
from .scaled_dot_product import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # d_model = total embedding dimension (e.g. 512)
        # num_heads = how many heads to split into (e.g. 8)
        # d_k = dimension per head (d_model/num_heads)
        super().__init__()
        assert d_model % num_heads == 0, "Total embedding dimension must be divisible by number of heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model//self.num_heads

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
    
   