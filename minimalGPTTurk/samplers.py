"""
Modelden metin oluşturmak için temel 'sampler'ları içerir.
Bu sampler'ların her biri modelden çıkan logitleri alır ve tahmin edilen tokenları döndürür.
Eğitilen modeli sampler'lar ile birlikte nasıl kullanacağınıza dair örnek için inference.ipynb'i inceleyiniz.
"""
import torch
import torch.nn.functional as F

class Sampler:
    """
    Temel sampler sınıfı. Diğer sampler'ların üzerinde kurulu olduğu iskeleti sağlar.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Tüm sampler alt-classları bir sample methodu tanımlamalıdır.")

class RandomSampler(Sampler):
    """
    Rastgele bir token seçer. Kıyaslama amaçlı kullanılabilir.
    """
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self.apply_temperature(logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

class GreedySampler(Sampler):
    """
    En basit sampler olan GreedySampler en yüksek logit değerine sahip tokenı seçer.
    """

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self.apply_temperature(logits)
        return torch.argmax(logits, dim=-1, keepdim=True) # (batch_size, 1) şeklinde bir tensör döndürür.

class TopKSampler(Sampler):
    """
    Top-k sampler, en yüksek olasılığa sahip k token arasından seçim yapar.
    """
    def __init__(self, temperature: float = 1.0, k: int = 50):
        super().__init__(temperature)
        self.k = k

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self.apply_temperature(logits)
        # sadece en yüksek değere sahip k kadar tokeni seçer.
        top_logits, top_indices = torch.topk(logits, self.k)
        probs = F.softmax(top_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(dim=-1, index=idx_next)
    
class NucleusSampler(Sampler):
    """
    Nucleus (Top-p) sampling, kümülatif olasılığın p olasılık değerini geçmediği tokenlar arasından seçim yapar.
    """
    def __init__(self, temperature: float = 1.0, p: float = 0.9):
        super().__init__(temperature)
        self.p = p

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self.apply_temperature(logits)
        # logitlerin olasılığını al
        probs = F.softmax(logits, dim=-1)
        # olasılığa göre sırala (azalan sıra)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        # kümülatif olasılığı al
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # p'den büyük olan indexten sonrasını sil.
        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0  # her zaman en yüksek olasılığı tut.

        # yukarıda tanımladığımız indexlemeyi kullanalım
        sorted_probs[sorted_indices_to_remove] = 0.0

        # olasılıkları normalize edelim ve next_idxlerle beraber döndürelim.
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        idx_next = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(dim=-1, index=idx_next)