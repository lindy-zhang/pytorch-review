import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "transformer"))
from tiny_gpt import TinyGPT

import math

import torch
import torch.nn.functional as F
from torch import Tensor

TEXT = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune "
    "or to take arms against a sea of troubles "
) * 20

def build_dataset(text: str, block_size: int):
    chars = sorted(set(text)) # sorted list of unique characters
    stoi = {char: i for i, char in enumerate(chars)} # str to int
    itos = {i: char for i, char in enumerate(chars)} # int to str
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # split into training/validation datasets
    n = int(0.9*len(data)) # index marking 90% split point
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, stoi, itos

def get_batch(data: Tensor, block_size: int, batch_size: int):
    ix = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x, y

if __name__ == "__main__":
    # Test build_dataset and get_batch
    train_data, val_data, stoi, itos = build_dataset(TEXT, block_size=32)
    print("vocab size:", len(stoi))
    print("train tokens:", len(train_data), "val tokens:", len(val_data))

    x, y = get_batch(train_data, block_size=32, batch_size=4)
    print("x shape:", x.shape, "y shape:", y.shape)
    print("sample x[0]:", "".join(itos[i.item()] for i in x[0]))
    print("sample y[0]:", "".join(itos[i.item()] for i in y[0]))