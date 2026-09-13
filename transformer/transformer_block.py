import torch
from torch import Tensor
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "attention"))
from multi_head_attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff, bias=True),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model, bias=True))
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=True, bias=True)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=True, bias=True)
        self.ffn = FeedForward(d_model, d_ff)
    
    def forward(self, x, causal: bool=True) ->  Tensor:
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.ffn(self.ln2(x))
        return x

if __name__ == "__main__":
    # Transformer Block test
    torch.manual_seed(0)
    batch, seq_len, d_model, num_heads, d_ff = 2, 12, 128, 8, 512

    block = TransformerBlock(d_model, num_heads, d_ff)
    x = torch.randn(batch, seq_len, d_model)
    out = block(x)

    print("output shape:", out.shape)
    assert out.shape == x.shape

    n_params = sum(p.numel() for p in block.parameters())
    print(f"param count: {n_params:,}")