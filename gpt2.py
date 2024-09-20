"""
GPT-2 modelinin implementasyonunu içerir.
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from modeling import LayerNorm, GELU, MultiHeadAttention

class FFN(nn.Module):
    """
    GPT-2 modelinin yapısındaki MLP bloklarını implemente eder.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=cfg.bias),
            GELU(),
            nn.Linear(4*cfg.n_embd, cfg.n_embd)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
class Block(nn.Module):
    """
    GPT-2 modelinin bir transformer bloğunu implemente eder.
    """
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(normalized_shape=cfg.n_embd, learn_scale_shift=cfg.bias)
        self.att = MultiHeadAttention(cfg)
        self.ln_2 = LayerNorm(normalized_shape=cfg.n_embd, learn_scale_shift=cfg.bias)
        self.ffn = FFN(cfg)
    
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x)) # x ile ekleyip forward eklemek shortcut ekliyor.
        x = x + self.ffn(self.ln_2(x))
        return x