import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken 
from dataclasses import dataclass
import time 
import math 
import inspect 
import numpy as np 
import os 

#----------------------Devices-----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed(1337)
    
#---------------------Parameters---------------------
@dataclass
class GPTConfig():    
    vocab_size = 50304
    n_embd = 768 
    num_layer = 12
    block_size = 1024
    n_head = 12

#---------------------Model Architecture---------------------
class AttentionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE  = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
class BLOCK(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = AttentionNet(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h   = nn.ModuleList([BLOCK(config) for n in range(self.config.num_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd),            
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE'):
                std *= (2 * self.config.num_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean= 0.0, std= std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean= 0.0, std= 0.02)           
        
    def forward(self, idx, target = None):
            B, T = idx.shape
            index = torch.arange(0, T, device=idx.device)
            
            pos_emb = self.transformer.wpe(index) 
            tok_emb = self.transformer.wte(idx)  
            x =  pos_emb + tok_emb      
            
            for H in self.transformer.h:
                x = H(x)

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if target is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

#-------------------------Data Loader-----------------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        data_root = r"F:\works\A-important\A-neurals\Vortex-Language-Models\GPT-2 From scratch\edu_fineweb10B"  
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x.to(device), y.to(device)

#-----------------model initializing and move to device-------------- 
model = GPT2(GPTConfig())
model.to(device)
torch.set_float32_matmul_precision('high')
print(f'\nDevice Used To Load MODEL is : {device}\n')
                
#---------------------Custom Training loop setings-------------------------
num_iter = 100
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073   

def get_lr(it): 
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps 
    if it > max_steps:
        return min_lr 
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)

total_batch_size = 524288
B = 8
T = 1024
assert total_batch_size % (B*T) == 0, "make sure total batch size is divisible by B*T"
grad_accum_steps = total_batch_size // (B*T)
print(f"Total Batch size is {total_batch_size}")
print(f"Gram accumulation is {grad_accum_steps}")

train_loader = DataLoaderLite(B, T, 'train')
val_loader = DataLoaderLite(B, T, 'val')

log_dir = r"F:\works\A-important\A-neurals\Vortex-Language-Models\GPT-2 From scratch\logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.txt")

#-------------------------------Gpt2 official tokenizer--------------------------
enc = tiktoken.get_encoding("gpt2")
val_lossi, trainl_lossi = [], []

#---------------------Custom Training loop-------------------------

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    #--------------------Evaluate validation loss---------------------
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f"validation loss: {val_loss_accum.item():.4f}")
        val_lossi.append(val_loss_accum.item())
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        if step > 0 and (step % 5000 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            torch.save(checkpoint, checkpoint_path)

    #-----------------------paragraph------------------------------------------
    if (step > 0 and step % 250 == 0) or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("generally LLM means")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"\n---------------\nsample {i}: {decoded}\n-----\n")

    
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    trainl_lossi(loss_accum.item())
    print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")