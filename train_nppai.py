import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import os
import glob

# 1. Definicja architektury w PyTorch (Dokładnie takiej samej jak w C++)
class NppAILayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.rms_attn = nn.RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.rms_ffn = nn.RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, mask):
        residual = x
        x_norm = self.rms_attn(x)
        
        q = self.wq(x_norm)
        k = self.wk(x_norm)
        v = self.wv(x_norm)
        
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        
        out = self.wo(attn_out)
        x = residual + out
        
        residual = x
        x_norm = self.rms_ffn(x)
        
        gate = F.silu(self.w1(x_norm))
        up = self.w2(x_norm)
        ffn_out = self.w3(gate * up)
        
        x = residual + ffn_out
        return x

class NppAIModel(nn.Module):
    def __init__(self, vocab_size=256, dim=64, hidden_dim=128, n_layers=2, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.pos_embeddings = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList([NppAILayer(dim, hidden_dim) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        B, T = tokens.size()
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
        x = self.tok_embeddings(tokens) + self.pos_embeddings(pos)
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=tokens.device)).view(1, T, T)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        logits = self.output(x)
        return logits

# 2. Funkcja do eksportowania wag modelu do formatu binarnego (.nppai)
def export_to_bin(model, filepath):
    print(f"Eksportowanie wag do {filepath}...")
    with open(filepath, 'wb') as f:
        # Nagłówek
        f.write(struct.pack('i', model.dim))
        f.write(struct.pack('i', model.hidden_dim))
        f.write(struct.pack('i', model.n_layers))
        f.write(struct.pack('i', model.max_seq_len))
        f.write(struct.pack('i', model.vocab_size))

        def write_tensor(tensor, is_linear=False):
            if is_linear:
                data = tensor.transpose(0, 1).contiguous().detach().cpu().numpy().flatten()
            else:
                data = tensor.detach().cpu().numpy().flatten()
            f.write(data.tobytes())

        # Wagi Embeddingu
        write_tensor(model.tok_embeddings.weight)
        write_tensor(model.pos_embeddings.weight)

        # Wagi Warstw
        for layer in model.layers:
            write_tensor(layer.rms_attn.weight)
            write_tensor(layer.wq.weight, is_linear=True)
            write_tensor(layer.wk.weight, is_linear=True)
            write_tensor(layer.wv.weight, is_linear=True)
            write_tensor(layer.wo.weight, is_linear=True)
            
            write_tensor(layer.rms_ffn.weight)
            write_tensor(layer.w1.weight, is_linear=True)
            write_tensor(layer.w2.weight, is_linear=True)
            write_tensor(layer.w3.weight, is_linear=True)

        # Wagi Wyjściowe
        write_tensor(model.norm.weight)
        write_tensor(model.output.weight, is_linear=True)
        
    print("Zakończono pomyślnie!")

# 3. Przygotowanie Datasetu w formacie "Instruct"
def load_dataset(folder_path="datasets/"):
    text = ""
    print(f"Szukanie datasetów uczących w folderze: {folder_path}...")
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} nie istniał, został utworzony. Wrzuć tam pliki .txt")
        return ""

    files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not files:
        print(f"Brak plików .txt w folderze {folder_path}!")
        return ""
        
    print(f"Znaleziono {len(files)} plików instruktażowych. Wczytywanie...")
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as file:
                text += file.read() + "\n\n"
        except Exception as e:
            print(f"Błąd czytania pliku {f}: {e}")
            
    print(f"Wczytano pomyślnie. Długość tekstu: {len(text)} znaków.")
    return text

# Prosty Tokenizer bajtowy (Zgodny w 100% z C++)
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        
    def encode(self, s):
        return list(s.encode('utf-8'))
        
    def decode(self, l):
        return bytes(l).decode('utf-8', errors='replace')

if __name__ == "__main__":
    print("Inicjalizacja środowiska trenowania NppAI...")
    
    # 1. Wczytanie kodu C++ wtyczki jako datasetu
    dataset_text = load_dataset()
    if len(dataset_text) < 100:
        print("Błąd: Za mało kodu w folderze src/ do trenowania.")
        exit(1)
        
    tokenizer = ByteTokenizer()
    data = torch.tensor(tokenizer.encode(dataset_text), dtype=torch.long)
    
    print(f"Rozmiar słownika (znaki unikalne): {tokenizer.vocab_size}")
    
    # 2. Inicjalizacja modelu z dopasowanym vocab_size
    model = NppAIModel(vocab_size=tokenizer.vocab_size, dim=64, hidden_dim=128, n_layers=4, max_seq_len=256)
    
    # 3. Parametry treningu
    learning_rate = 1e-3
    batch_size = 16 # Zmniejszamy by szybciej trenować na CPU
    block_size = 256 # Zwiększamy kontekst ze 128 na 256 dla dłuższych kodów
    max_iters = 2500 # Wystarczająca ilość kroków dla tego rozmiaru datasetu
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def get_batch():
        # Wybieramy losowe miejsca w kodzie
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix]) # Wejście (np. "int main() ")
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Oczekiwane wyjście (np. "nt main() {")
        return x, y

    print("Trenowanie modelu NppAI na kodzie C++...")
    model.train()
    
    for iter in range(max_iters):
        xb, yb = get_batch()
        
        # Forward pass
        logits = model(xb)
        
        # PyTorch spodziewa się kształtu (B*T, C) do obliczenia błędu (Cross Entropy)
        B, T, C = logits.shape
        logits_reshaped = logits.view(B*T, C)
        targets = yb.view(B*T)
        
        loss = F.cross_entropy(logits_reshaped, targets)
        
        # Backward pass (Aktualizacja wag)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Krok {iter}/{max_iters} | Błąd (Loss): {loss.item():.4f}")

    print("Trening zakończony!")
    
    # Eksportujemy model do formatu czytelnego dla naszej wtyczki C++
    os.makedirs("models", exist_ok=True)
    export_to_bin(model, "models/NppAI-model-v1.nppai")
