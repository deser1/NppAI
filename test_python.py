import torch
import torch.nn as nn
import struct
import os

# --- ARCHITEKTURA MODELU ---
class TransformerLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.rms_attn = nn.RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.rms_ffn = nn.RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x):
        # Self Attention
        h = self.rms_attn(x)
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)
        
        T = q.size(1)
        # Obliczenie attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = self.wo(out)
        x = x + out
        
        # Feed Forward (SiLU)
        h = self.rms_ffn(x)
        ffn_out = self.w2(torch.nn.functional.silu(self.w1(h)) * self.w3(h))
        x = x + ffn_out
        
        return x

class NppAIModel(nn.Module):
    def __init__(self, vocab_size=256, max_seq_len=4096, dim=256, hidden_dim=512, n_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList([TransformerLayer(dim, hidden_dim) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.head(x)
        return logits

def generate(model, prompt, max_new_tokens=50, temperature=0.35, top_k=3):
    model.eval()
    idx = torch.tensor([list(prompt.encode('utf-8'))], dtype=torch.long)
    
    print("Prompt:", prompt, end='')
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.max_seq_len:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] # Ostatni token
            
            # Top-K
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax + Temperature
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            
            # Sampling
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Detokenizacja i wypisanie
            token_val = idx_next.item()
            if token_val == 0:
                break
                
            try:
                char = bytes([token_val]).decode('utf-8')
                print(char, end='', flush=True)
            except:
                print('?', end='', flush=True)
                
            idx = torch.cat((idx, idx_next), dim=1)
            
    print("\n\nGenerowanie w Pythonie zakończone.")

# Załadujmy ostatnio trenowany checkpoint
model_path = "models/NppAI-model-v1.nppai"
if not os.path.exists(model_path):
    print(f"Brak pliku {model_path}! Upewnij się, że podałeś prawidłową ścieżkę do wagi modelu.")
else:
    print("Ładowanie modelu C++ do Pythona w celu walidacji (symulacja formatu C++)...")
    
    with open(model_path, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        hidden_dim = struct.unpack('i', f.read(4))[0]
        n_layers = struct.unpack('i', f.read(4))[0]
        max_seq_len = struct.unpack('i', f.read(4))[0]
        vocab_size = struct.unpack('i', f.read(4))[0]
        
        print(f"Dim: {dim}, Hidden: {hidden_dim}, Layers: {n_layers}")
        
        model = NppAIModel(vocab_size, max_seq_len, dim, hidden_dim, n_layers)
        
        def load_tensor(shape):
            num_elements = 1
            for s in shape: num_elements *= s
            data = struct.unpack(f'{num_elements}f', f.read(4 * num_elements))
            return torch.tensor(data, dtype=torch.float32).view(shape)
            
        model.token_emb.weight.data = load_tensor([vocab_size, dim])
        model.pos_emb.weight.data = load_tensor([max_seq_len, dim])
        
        for i in range(n_layers):
            model.layers[i].rms_attn.weight.data = load_tensor([dim])
            model.layers[i].wq.weight.data = load_tensor([dim, dim]).T
            model.layers[i].wk.weight.data = load_tensor([dim, dim]).T
            model.layers[i].wv.weight.data = load_tensor([dim, dim]).T
            model.layers[i].wo.weight.data = load_tensor([dim, dim]).T
            
            model.layers[i].rms_ffn.weight.data = load_tensor([dim])
            model.layers[i].w1.weight.data = load_tensor([dim, hidden_dim]).T
            model.layers[i].w3.weight.data = load_tensor([dim, hidden_dim]).T
            model.layers[i].w2.weight.data = load_tensor([hidden_dim, dim]).T
            
        model.norm.weight.data = load_tensor([dim])
        model.head.weight.data = load_tensor([dim, vocab_size]).T

    print("\n--- TEST GENEROWANIA (Python) ---")
    prompt = "[USER]: Prosta strona html\n[AI]:\n"
    generate(model, prompt)
