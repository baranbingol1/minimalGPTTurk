"""
Oluşturduğunuz modelleri eğitmek için training scripti.
Minimal bir implementasyonu vardır ve modelden bağımsızdır.
https://github.com/karpathy/nanoGPT/blob/master/train.py bu dosyadan ciddi oranda esinlenilmiştir.
Daha da anlaşılabilir olması açısında DDP ve Gradient Accumulation özellikleri eklenmemiştir.
Eğer ciddi bir training yapmak istiyorsanız örneğin 8xGPU ile çok büyük bir veri üzerinde eğitim yapmak gibi
Yukarıdaki referansı kullanmanız daha iyi olacaktır.
Bu implementasyon sadece tek bir GPU üzerinde çalışır ancak yine de 
torch.compile ve mixed-precision gibi eğitimi hızlandıran faktörler eklenmiştir.
"""

import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext

### model importu
from gpt2 import GPT2Model, get_gpt2_cfg

####### PARAMETRELER #######
gpt2_cfg = get_gpt2_cfg('very_tiny', vocab_size=141) # buraya kullandığımız tokenizer'ın vocab_size'ını girmeliyiz.
model = GPT2Model(gpt2_cfg)
batch_size = 256 # verinizi tokenize ettiğiniz zamanki batch_size ile aynı olması lazım

block_size = gpt2_cfg.block_size
do_compile = False # True yapmanız önerilir şuan windowsta desteklenmediği için False yaptım :(

# Verinizin bulunduğu yer. Eğer kendi veri setinizle eğitim yapacaksanız nasıl hazırlanması gerektiğini öğrenmek için data/turcke_siirler/prepare.ipynb bakınız.
dataset = 'turkce_siirler' 
out_dir = 'out' # checkpoint dosyasının gideceği yer

eval_iters = 200
max_iters = 5000 # ne kadar training adımı olacağı

eval_interval = 1000 # ne kadar training adımında bir evaluation yapacağı (bunu zaman aldığından sık yapmanızı önermem).
log_interval = 100 # ne kadar training adımında bir bilgileri ekrana yansıtacağı

learning_rate = 6e-4
weight_decay = 1e-1

decay_lr = True # learning rate decay kullanılsın mı 
lr_decay_iters = max_iters
min_lr = 6e-5 # learning rate decay'in yakınsayacağı learning rate
warmup_iters = 100

device = 'cuda' # 'cpu' veya macbook kullanıyorsanız 'mps' (birden fazla gpu varsa 'cuda:0', 'cuda:1' gibi spesifik seçebilirsiniz.)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
dtype_of_data = np.uint8 # prepare.ipynb'de kullanılan dtype.
ctx_dtype = torch.bfloat16 if dtype =='bfloat16' else torch.float16
ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type=device, dtype=ctx_dtype)
############################

data_pth = os.path.join('data', dataset)

# modeli eğitirken veriyi alacağımız fonksiyon.
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_pth, 'train.bin'), dtype=dtype_of_data, mode='r')
    else:
        data = np.memmap(os.path.join(data_pth, 'val.bin'), dtype=dtype_of_data, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # cpu ve gpu arasında asenkron transfer sağlaması için pin_memory ekliyoruz
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# model'i uygun device'a taşıyıp compile ediyoruz. (compile etmek modeli çok daha hızlandırır)
# torch.compile, pytorch >= 2.0 ile geldi.

model.to(device)
if do_compile:
    model = torch.compile(model)

# eğer dtype float16 ise cuda yoktur böylece enabled=False olur ve torch.amp devreye girmez.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# yeterince doğru bir loss hesaplar.
# tüm batchler üzerinde yapmak uzun süreceğinden böyle bir yakınsama yolu tercih edilir
# sadece validation da kullanılmak üzere yazıldı
@torch.inference_mode()
def estimate_loss():
    out = {}
    perps = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        mean_loss = losses.mean()
        out[split] = mean_loss
        perps[split] = torch.exp(mean_loss)
    model.train()
    return out, perps

def get_lr(it):
    # warmup iterasyonları bitene kadar warmup lrsi
    if it < warmup_iters:  
        return learning_rate * it / warmup_iters
    # eğer lr_decay_iters den daha büyük bir iterasyondaysak
    # min_lr döndürülür. varsayılan olarak max_iter ile lr_decay_iter eşittir.
    # o yüzden tam olarak son iterasyonda learning_rate -> min_lr olmuş olacak.
    # ancak isterseniz yukarıdaki parametrelerden değiştirebilirsiniz tabii ki.
    if it > lr_decay_iters:
        return min_lr
    # onun dışında learning rate "cosine decay" ile azalıyor.
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loopundan önce optimizer'ı ayarlıyoruz
# 2d olan tüm parametre grupları weight decay'e uğrayacak diğerleri ise uğramayacak
param_dict = {pn: p for pn, p in model.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)

# training loopu
# klasik bir pytorch training loop boilerplate'i olduğu için yorumlanmamıştır.
X, Y = get_batch('train')
local_iter_num = 0
iter_num = 0
best_val_loss = 1_000_000
os.makedirs(out_dir, exist_ok=True)
while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses, perps = estimate_loss()
        print(f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity {perps['train']:.4f}, val perplexity {perps['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"checkpont {out_dir}'a kaydediliyor..")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    with ctx:
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
    
    X, Y = get_batch('train')
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # ekrana bilgi verme işlemleri
    if iter_num % log_interval == 0:
        if local_iter_num >= 5: # training başlarında log yapmıyoruz.
            print(f"iter {iter_num}: loss {loss:.4f}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break