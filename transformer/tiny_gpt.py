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

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        # Repeat the predict-append cycle max_new_tokens times
        for _ in range(max_new_tokens):
            logits = self(idx)
            last_logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(last_logits, dim=1)
            # Generate next token
            next_token = torch.multinomial(probs, num_samples=1)
            # Append next token
            idx = torch.cat([idx, next_token], dim=1)
        return idx

if __name__ == "__main__":
    # Test forward
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


    # Test generate
    generated = model.generate(idx[:1, :4], max_new_tokens=10)
    print("generated shape:", generated.shape)
    assert generated.shape == (1, 14)
    print("OK generate")