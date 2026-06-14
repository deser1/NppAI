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
        # Self-Attention
        self.rms_attn = nn.RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        # Feed-Forward Network (SwiGLU)
        self.rms_ffn = nn.RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=False) # up
        self.w3 = nn.Linear(hidden_dim, dim, bias=False) # down

    def forward(self, x):
        # Implementacja matematyki Transformera w PyTorch
        residual = x
        
        # --- Self-Attention ---
        # Normalizacja
        x_norm = self.rms_attn(x)
        
        # Projekcje
        q = self.wq(x_norm)
        k = self.wk(x_norm)
        v = self.wv(x_norm)
        
        # Obliczanie atencji: softmax(Q * K^T / sqrt(d)) * V
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        
        # Projekcja wyjściowa
        out = self.wo(attn_out)
        
        # Residual connection
        x = residual + out
        
        # --- Feed-Forward Network ---
        residual = x
        x_norm = self.rms_ffn(x)
        
        # SwiGLU: SiLU(x * W1) * (x * W2) * W3
        gate = F.silu(self.w1(x_norm))
        up = self.w2(x_norm)
        ffn_out = self.w3(gate * up)
        
        # Residual connection
        x = residual + ffn_out
        
        return x

class NppAIModel(nn.Module):
    def __init__(self, vocab_size=256, dim=64, hidden_dim=128, n_layers=2, n_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([NppAILayer(dim, hidden_dim) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.tok_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits

# 2. Funkcja do eksportowania wag modelu do formatu binarnego (.nppai)
def export_to_bin(model, filepath):
    print(f"Eksportowanie wag do {filepath}...")
    with open(filepath, 'wb') as f:
        # Nagłówek (5 x int32)
        f.write(struct.pack('i', model.dim))
        f.write(struct.pack('i', model.hidden_dim))
        f.write(struct.pack('i', model.n_layers))
        f.write(struct.pack('i', model.n_heads))
        f.write(struct.pack('i', model.vocab_size))

        # Funkcja pomocnicza do zapisu tensora (1D)
        def write_tensor(tensor):
            data = tensor.detach().cpu().numpy().flatten()
            f.write(data.tobytes())

        # Wagi Embeddingu
        write_tensor(model.tok_embeddings.weight)

        # Wagi Warstw
        for layer in model.layers:
            write_tensor(layer.rms_attn.weight)
            write_tensor(layer.wq.weight)
            write_tensor(layer.wk.weight)
            write_tensor(layer.wv.weight)
            write_tensor(layer.wo.weight)
            
            write_tensor(layer.rms_ffn.weight)
            write_tensor(layer.w1.weight)
            write_tensor(layer.w2.weight)
            write_tensor(layer.w3.weight)

        # Wagi Wyjściowe
        write_tensor(model.norm.weight)
        write_tensor(model.output.weight)
        
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
    model = NppAIModel(vocab_size=tokenizer.vocab_size, dim=64, hidden_dim=128, n_layers=4, n_heads=4)
    
    # 3. Parametry treningu
    learning_rate = 1e-3
    batch_size = 4 # Ilość fragmentów kodu analizowanych jednocześnie
    block_size = 64 # Długość fragmentu kodu (kontekst)
    max_iters = 1000 # Ilość kroków treningu
    
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
