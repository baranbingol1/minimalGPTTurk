import torch

def generate_text_simple(model, idx: torch.Tensor, max_new_tokens: int, context_size: int):
    # idx -> (batch_size, token_size) boyutunda olmalı.
    for _ in range(max_new_tokens):
        
        # context_size'a uygun crop
        # örneğin 512 token verildiyse ama modelimiz context length'i 256 ise
        # son 256 token'ı alıcaktır.
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # inference için mini-optimizasyon sadece son token'a softmax yapıyoruz
        # (batch_size, n_tokens, vocab_size) boyutundan -> (batch_size, vocab_size) boyutuna
        logits = logits[:, -1, :]  

        probs = torch.softmax(logits, dim=-1)  # boyutu (batch_size, vocab_size)

        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # boyutu (batch_size, 1)

        # oluşturulan token'ı ekleyip modele geri atıyoruz.
        idx = torch.cat((idx, idx_next), dim=1)  # boyutu (batch_size, n_tokens+1)

    return idx