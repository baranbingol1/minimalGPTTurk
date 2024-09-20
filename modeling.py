"""
Bu dosyada GPT modelleri için gereken yapı taşları implemente edilmiştir.
Referanslar:
1) https://nn.labml.ai/
2) https://github.com/karpathy/nanoGPT
3) https://github.com/rasbt/LLMs-from-scratch
"""


from typing import Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """LayerNormalization (https://arxiv.org/abs/1607.06450) implementasyonu.

    BatchNormalization'a göre ana avantajları batch_size'a bağlı olmaması ve distributed training için daha uygun olması.
    Ana farkları ise BN'nin batch dimension boyunca normalizasyon yapması LN'nin ise feature dimension'ı boyunca normalizasyon yapmasıdır.
    Dolayısıyla BN batch boyutuna bağlı kalmakta ve batch boyutu modelin performansını etkilemektedir.

    Argümanlar:
        normalized_shape: batch_size boyutu hariç girdinin boyutu. Tipi [int, int listesi veya torch.Size] olmalıdır.
        eps: nümerik hesaplamayı olabilir kılmak için küçük bir değer. varsayılan : 1e-5 (float)
        learn_scale_shift: True ise normalize edilmiş değerlere scaling ve shifting uygular. varsayılan : True (boolean)
    """
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], eps: float = 1e-5, learn_scale_shift: bool = True):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        
        assert isinstance(normalized_shape, torch.Size), "normalized_shape'in tipi [int, int listesi veya torch.Size] olmalıdır"

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.learn_scale_shift = learn_scale_shift

        if self.learn_scale_shift:
            # γ ve β öğrenilebilir parametreleri.
            self.γ = nn.Parameter(torch.ones(normalized_shape))
            self.β = nn.Parameter(torch.zeros(normalized_shape))


    def forward(self, x: torch.Tensor):
        """x: [seq_len, batch_size, feats] şeklinde bir girdi vektörü olmalıdır."""
        # ortalamanın hesaplanacağı boyutlar
        dims = [-(i + 1) for i in range(len(self.normalized_shape))] 
        
        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x**2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean**2 # var[x] = E[x^2] - E[x]^2

        x_norm = (x - mean) / torch.sqrt(var+self.eps)

        if self.learn_scale_shift:
            x_norm = self.γ*x_norm + self.β
        
        return x_norm
    
# gpt-2 makalesinde aktivasyon fonksiyonu olarak GELU kullanılmıştır.
# GLU ve varyantları transformerlar için daha uygundur. Aşağıdaki makale bunu kanıtlamaktadır.
# https://arxiv.org/abs/2002.05202
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """GELU analitik olarak GELU(x) = x * Φ(x) şeklinde hesaplanabilir. (Φ, gaussian cumulative distribution function olmak üzere)
        Ancak bunu yapmak hesaplama bakımından masraflıdır dolayısıyla tanh yakınsamasını kullanmak daha doğrudur. 
        Aşağıda GELU fonksiyonu için tanh yakınsaması implemente edilmiştir.
        Detaylar için : https://arxiv.org/pdf/1606.08415
        """
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.004715 * torch.pow(x,3))))

# flash_attentionun olup olmadığını kontrol eder (flash attention yalnızca pytorch >=2.0 için var)
FLASH_ATTENTION_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mekanizmasının implementasyonu.

    Multi-Head Attention, Transformer mimarisinde kullanılan temel bir yapı taşıdır. 
    Birden fazla attention başlığı (head) kullanarak attention skoru hesaplar. 
    Her head, kendi attention matrisini öğrenir ve çıktılar bu head'ler arasında birleştirilir (concat).

    Argümanlar:
        cfg: Modelin yapılandırma dosyası. Aşağıdaki örnek bir cfg verilmiştir (GPT-2-Small yapısına eş değer bir yapı döndürür)

            >>> @dataclass
            >>> class GPTConfig:
            >>>    block_size: int = 1024
            >>>    vocab_size: int = 50257 
            >>>    n_layer: int = 12
            >>>    n_head: int = 12
            >>>    n_embd: int = 768
            >>>    dropout: float = 0.0
            >>>    bias: bool = True

    Not:
        Flash attention yalnızca PyTorch >= 2.0 sürümünde kullanılabilir. Eğer daha eski bir sürümde çalışıyorsanız, flash attention kullanılamaz ve hata mesajı döner.
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "Headlere eşit ağırlık dağılabilmesi için [n_embd / n_head] tam sayı olmalıdır."
        # şuanlık pytorch < 2.0 için desteğimiz yok. ileride eklenebilir.
        if not FLASH_ATTENTION_AVAILABLE:
            raise Exception("Flash attention kullanılabilir değil. Lütfen PyTorch'u >=2.0 sürümüne güncelleyin.")
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.bias = cfg.bias
        self.dropout = cfg.dropout

        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd, bias=self.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)

        self.attn_dropout = nn.Dropout(self.dropout) # masked-attention için dropout
        self.resid_dropout = nn.Dropout(self.dropout) # artık bağlantılar için dropout

    def forward(self, x):
        B, T, C = x.size() # x'in şekli -> batch_size, seq_len, n_embd

        # Q,K,V matrislerini hesaplar ve attention için dimensionlarını değiştirir.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self.training model.train() ise 1 aksi halde model.eval() ise 0 olur.
        # internal bir attributedur yani nn.Module inherit almamızdan geliyor.
        y = F.scaled_dot_product_attention(q, k, v, 
                                           attn_mask=None, 
                                           dropout_p=self.dropout if self.training else 0, 
                                           is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B,T,C) # attention headlerini tekrardan concat eder.

        y = self.resid_dropout(self.c_proj(y))
        return y
