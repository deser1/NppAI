import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import os
import glob
import math
import sys

sys.stdout.reconfigure(encoding='utf-8')

# --- KONFIGURACJA SPRZĘTU (DirectML / CUDA / CPU) ---
try:
    import torch_directml
    dml_available = torch_directml.is_available()
except ImportError:
    dml_available = False

# WYMUSZAMY CPU ze względu na błędy DML OOM na dużym block_size
print("Wymuszanie użycia CPU ze względu na duży rozmiar okna kontekstowego (uniknięcie DML OOM)...")
device = torch.device("cpu")
dml_available = False

print(f"Używam urządzenia do trenowania: {device}")
# ----------------------------------------------------

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
        
    print(f"Znaleziono {len(files)} plikow instruktazowych. Wczytywanie...")
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as file:
                text += file.read() + "\n\n"
        except Exception as e:
            print(f"Blad czytania pliku {f}: {e}")
            
    print(f"Wczytano pomyslnie. Dlugosc tekstu: {len(text)} znakow.")
    return text

# Prosty Tokenizer bajtowy (Zgodny w 100% z C++)
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        
    def encode(self, s):
        return list(s.encode('utf-8'))
        
    def decode(self, l):
        return bytes(l).decode('utf-8', errors='replace')

def get_batch(split, data, block_size, batch_size):
    # Generowanie losowych indeksów początkowych
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train_model():
    global device
    print("Inicjalizacja środowiska trenowania NppAI...")
    
    # 1. Wczytanie kodu C++ wtyczki jako datasetu
    dataset_text = load_dataset()
    
    if len(dataset_text) < 100:
        print("Błąd: Za mało kodu do trenowania.")
        exit(1)
        
    tokenizer = ByteTokenizer()
    data = torch.tensor(tokenizer.encode(dataset_text), dtype=torch.long)
    
    print(f"Rozmiar słownika (znaki unikalne): {tokenizer.vocab_size}")
    
    # 2. Inicjalizacja modelu z dopasowanym vocab_size
    # Powiększona, zoptymalizowana architektura do zapamiętania kodu programistycznego
    model = NppAIModel(vocab_size=tokenizer.vocab_size, dim=256, hidden_dim=512, n_layers=4, max_seq_len=2048)
    
    # SYSTEM CHECKPOINTÓW (PAMIĘCI MODELU)
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print("\n[!] Znaleziono poprzednie wagi! Ładowanie nabytej wiedzy z checkpoint.pth...")
        # Wczytujemy zapisaną wcześniej wiedzę (State Dictionary) do modelu
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    else:
        print("\n[!] Brak poprzednich wag. Rozpoczynamy naukę od zera.")

    # Przenosimy model na odpowiednie urządzenie w sposób bezpieczny dla VRAM
    if dml_available:
        import torch_directml
        os.environ["DML_DISABLE_MEMORY_OPT"] = "1"
    
    try:
        model.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or dml_available:
            print(f"\n[!] Błąd pamięci GPU (DirectML): {e}")
            print("[!] Karta graficzna odrzuciła alokację pamięci (częsty problem ze zintegrowanymi kartami lub brakiem VRAM).")
            print("[!] Automatyczny powrót do trenowania na procesorze (CPU)...")
            device = torch.device("cpu")
            model.to(device)
        else:
            raise e
    
    # 3. Parametry treningu
    # Używamy wolniejszego uczenia, ale na większej ilości danych
    learning_rate = 5e-4
    max_iters = 3000 # Jeszcze więcej kroków
    block_size = 2048 # Zwiększone z 512 na 2048 by nauczyć model całego max_seq_len
    batch_size = 4 # Zmniejszone z 8 na 4 by zmieścić w RAM
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Trenowanie modelu NppAI...")
    model.train()
    
    for iter in range(max_iters):
        # Losujemy partię z całego zbioru, nie używamy x_full by model nie był zafiksowany na 1 przykładzie
        xb, yb = get_batch('train', data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass
        logits = model(xb)
        
        B, T, C = logits.shape
        logits_reshaped = logits.view(B*T, C)
        targets = yb.view(B*T)
        
        loss = F.cross_entropy(logits_reshaped, targets)
        
        # Backward pass (Aktualizacja wag)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 10 == 0 or iter == max_iters - 1:
            # Wymuszamy opróżnienie bufora Pythona, aby plik na dysku zawsze miał najświeższą wartość!
            print(f"Krok {iter}/{max_iters} | Błąd (Loss): {loss.item():.4f}", flush=True)
            
        # Zapisuj postęp do pliku checkpoint.pth co 50 kroków (zabezpieczenie przed uśpieniem)
        if iter > 0 and iter % 50 == 0:
            torch.save(model.state_dict(), checkpoint_path)
            
    print("\nTrening zakończony!")
    
    # Zapisujemy wiedzę (checkpoint) do dalszego trenowania w przyszłości
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Zapisano nabytą wiedzę (stan matematyczny) do {checkpoint_path}")
    
    # Eksportujemy model do formatu czytelnego dla naszej wtyczki C++
    export_to_bin(model, "models/NppAI-model-v1.nppai")

if __name__ == "__main__":
    train_model()
