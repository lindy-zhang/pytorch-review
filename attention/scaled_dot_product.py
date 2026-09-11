import math
import torch # PyTorch libary
from torch import Tensor

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool=False) -> Tensor:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    q: (batch, heads, seq_len_q, d_k)
    k: (batch, heads, seq_len_k, d_k)
    v: (batch, heads, seq_len_k, d_v)
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k) # shape: (batch, heads, seq_len_q, seq_len_k)
    # Causal masking
    if causal:
        seq_len_q, seq_len_k = scores.size(-2), scores.size(-1)
        mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool)
        mask = torch.tril(mask, 0)
        scores = scores.masked_fill(~mask, float("-inf"))
    # Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    # Weighted sum
    output = attention_weights @ v
    return output

if __name__ == "__main__":
    q = torch.randn(2, 4, 6, 16)
    k = torch.randn(2, 4, 6, 16)
    v = torch.randn(2, 4, 6, 16)
    out = scaled_dot_product_attention(q, k, v, causal=True)
    print(out.shape)