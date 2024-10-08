"""
GPT-2 modelinin implementasyonunu içerir.
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from modeling import LayerNorm, GELU, MultiHeadAttention

######## GPT-2 YAPI TAŞLARI ########

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
        # shortcut oluşturmak için x = x + işlem şeklinde yapıyoruz.
        x = x + self.att(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x
    
######## GPT-2 MODELİ ########

class GPT2Model(nn.Module):
    """GPT 2 modelinin implementasyonu
    Orijinal tensorflow implementasyonu ile karşılaştırma yapmak için :
    https://github.com/openai/gpt-2/blob/master/src/model.py bakabilirsiniz.
    Ayrıca bu linkteki 
    https://www.researchgate.net/figure/GPT-2-model-architecture-The-GPT-2-model-contains-N-Transformer-decoder-blocks-as-shown_fig1_373352176
    görselleştirmeye bakmakta modeli anlamak açısından yararlı olabilir
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd) # token embeddings
        self.wpe = nn.Embedding(self.cfg.block_size, self.cfg.n_embd) # positional embeddings
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.hidden_trf_blocks = nn.Sequential(
            *[Block(self.cfg) for _ in range(self.cfg.n_layer)]
        )
        self.ln_f = LayerNorm(self.cfg.n_embd)
        self.out_head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)

        # Weight tying yöntemi son linear layerımızla embedding layerımızdaki weightleri eşitliyoruz.
        # Bu yöntemin dil modellerinde performansı arttırdığı bilinmektedir
        # https://arxiv.org/pdf/1608.05859
        self.wte.weight = self.out_head.weight

        self.apply(self._init_weights)

        # gpt-2 makalesindeki gibi artık bağlantılara farklı bir weight initilization uyguluyoruz.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/((2*self.cfg.n_layer)**(1/2)))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor):
        device = idx.device
        b, t = idx.size() # batch ve tokenların uzunluğu.

        assert t <= self.cfg.block_size, f"{t} olan token boyutu uzunluğu, context-length olan {self.cfg.block_size} dan daha uzun."
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.hidden_trf_blocks(x)
        x = self.ln_f(x)
        
        logits = self.out_head(x)
        
        return logits
    
######## GPT-2 CONFİGLERİ ########

@dataclass
class GPT2Config:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True

def get_gpt2_cfg(model_size: Literal["very_tiny", "tiny", "small", "medium", "large", "xl"], vocab_size: int) -> GPT2Config:
    configs = {
        "very_tiny": {"n_layer": 3, "n_head": 3, "n_embd": 192, "block_size": 128}, # orijinal makalede olmayan uydurduğum minnak bir gpt-2
        "tiny": {"n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 512}, # yine uydurduğum biraz daha az minnak bir gpt-2
        "small":  {"n_layer": 12, "n_head": 12, "n_embd": 768},
        "medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
        "large":  {"n_layer": 36, "n_head": 20, "n_embd": 1280},
        "xl":     {"n_layer": 48, "n_head": 25, "n_embd": 1600}
    }
    
    model_config = configs.get(model_size.lower(), configs["small"])
    return GPT2Config(vocab_size=vocab_size, **model_config)