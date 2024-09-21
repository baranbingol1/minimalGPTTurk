"""
NOT : 

Testler ana klasörden
python -m unittest discover -s tests
komutuyla çalıştırılmalıdır.
"""

from gpt2 import GPT2Model, get_gpt2_cfg
from test_utils import generate_text_simple
import torch
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = tiktoken.encoding_for_model('gpt-4o')
model_cfg = get_gpt2_cfg(model_size='small', vocab_size=encoder.n_vocab)
model = GPT2Model(model_cfg).to(device)

sample_start_ctx = "Merhaba, ben "
encoded = encoder.encode(sample_start_ctx)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
encoded_tensor = encoded_tensor.to(device=device)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=model_cfg.block_size
)

generated_text = encoder.decode(out.squeeze(0).tolist())

# unit testler
assert len(generated_text) > len(sample_start_ctx), "Model ekstra token üretmeli"
assert generated_text.startswith(sample_start_ctx), "Üretilen tokenlar verilen context ile başlamalı"