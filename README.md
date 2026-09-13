# pytorch-review

GPT-style Transformer, implemented from scratch to review my PyTorch

## structure

- `attention/` — scaled dot-product attention, multi-head attention
- `transformer/` — transformer block, TinyGPT model
- `training/` — dataset, batching, training loop

## how to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 training/train_tiny_gpt.py
```
