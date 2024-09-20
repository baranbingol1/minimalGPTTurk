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
