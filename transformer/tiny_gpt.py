import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformer_block import TransformerBlock

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
    
    def forward(self, idx: Tensor) -> Tensor: 
        # idx = input (token IDs)
        batch, seq_len = idx.shape
        positions = torch.arange(seq_len)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

if __name__ == "__main__":
    # Test
    torch.manual_seed(0)
    vocab_size = 65
    model = TinyGPT(vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=2, max_seq_len=32)

    batch, seq_len = 4, 16
    idx = torch.randint(0, vocab_size, (batch, seq_len))
    logits = model(idx)

    print("logits shape:", logits.shape)
    assert logits.shape == (batch, seq_len, vocab_size)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")
    print("OK")
