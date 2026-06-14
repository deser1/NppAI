#include "NppAIEngine.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>

// --- TENSOR IMPLEMENTATION ---
Tensor::Tensor(std::vector<int> s) : shape(s) {
    int size = 1;
    for (int d : shape) size *= d;
    data.resize(size, 0.0f);
}

float& Tensor::at(int i) { return data[i]; }
float& Tensor::at(int r, int c) { return data[r * shape[1] + c]; }

// Podstawowe mnożenie macierzy O(N^3) (w przyszłości do optymalizacji np. przez OpenMP/SIMD)
Tensor Tensor::matmul(const Tensor& a, const Tensor& b, bool transposeB) {
    int a_rows = a.shape[0];
    int a_cols = a.shape[1];
    int b_rows = transposeB ? b.shape[1] : b.shape[0];
    int b_cols = transposeB ? b.shape[0] : b.shape[1];
    
    Tensor result({a_rows, b_cols});
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; k++) {
                float b_val = transposeB ? b.data[j * b.shape[1] + k] : b.data[k * b.shape[1] + j];
                sum += a.data[i * a.shape[1] + k] * b_val;
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

void Tensor::applySiLU() {
    for (auto& val : data) {
        val = val / (1.0f + std::exp(-val)); // x * sigmoid(x)
    }
}

void Tensor::applyRMSNorm(const Tensor& weight) {
    // Obliczanie średniej kwadratowej
    float ss = 0.0f;
    for (float val : data) ss += val * val;
    ss /= data.size();
    ss += 1e-5f; // epsilon
    ss = 1.0f / std::sqrt(ss);

    // Normalizacja i skalowanie
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = (data[i] * ss) * weight.data[i];
    }
}

void Tensor::readFromFile(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
}

// --- ENGINE IMPLEMENTATION ---
NppAIEngine::NppAIEngine() {}
NppAIEngine::~NppAIEngine() {}

bool NppAIEngine::loadModel(const std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Nie udalo sie otworzyc pliku modelu: " << modelPath << std::endl;
        return false;
    }

    // Wczytywanie nagłówka (konfiguracji)
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&hidden_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&n_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&max_seq_len), sizeof(int));
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));

    // Inicjalizacja i wczytywanie wag
    tokenEmbeddingTable = Tensor({vocab_size, dim});
    tokenEmbeddingTable.readFromFile(file);

    posEmbeddingTable = Tensor({max_seq_len, dim});
    posEmbeddingTable.readFromFile(file);

    layers.clear();
    for (int i = 0; i < n_layers; i++) {
        TransformerLayer layer;
        layer.rmsAttn = Tensor({dim}); layer.rmsAttn.readFromFile(file);
        layer.wQ = Tensor({dim, dim}); layer.wQ.readFromFile(file);
        layer.wK = Tensor({dim, dim}); layer.wK.readFromFile(file);
        layer.wV = Tensor({dim, dim}); layer.wV.readFromFile(file);
        layer.wO = Tensor({dim, dim}); layer.wO.readFromFile(file);
        
        layer.rmsFFN = Tensor({dim}); layer.rmsFFN.readFromFile(file);
        layer.wGate = Tensor({dim, hidden_dim}); layer.wGate.readFromFile(file);
        layer.wUp = Tensor({dim, hidden_dim}); layer.wUp.readFromFile(file);
        layer.wDown = Tensor({hidden_dim, dim}); layer.wDown.readFromFile(file);
        
        layers.push_back(layer);
    }

    outputRMSNorm = Tensor({dim});
    outputRMSNorm.readFromFile(file);
    
    outputClassifier = Tensor({dim, vocab_size});
    outputClassifier.readFromFile(file);

    file.close();
    return true;
}

std::vector<int> NppAIEngine::tokenize(const std::string& text) {
    // Tokenizacja bajtowa zgodna z UTF-8
    std::vector<int> tokens;
    for (char c : text) tokens.push_back((int)(unsigned char)c);
    return tokens;
}

std::string NppAIEngine::detokenize(const std::vector<int>& tokens) {
    std::string text;
    for (int t : tokens) {
        if (t >= 0 && t < 256) {
            text += (char)(unsigned char)t;
        }
    }
    return text;
}

Tensor NppAIEngine::forward(const std::vector<int>& inputTokens) {
    int seq_len = inputTokens.size();
    if (seq_len == 0 || dim == 0) return Tensor({1, vocab_size});
    
    // Zabezpieczenie przed przekroczeniem kontekstu
    int T = std::min(seq_len, max_seq_len);
    
    // 1. Embedding dla całej sekwencji T
    Tensor x({T, dim});
    for(int pos = 0; pos < T; pos++) {
        int token = inputTokens[seq_len - T + pos]; // Bierzemy T ostatnich tokenów
        for(int i=0; i<dim; i++) {
            x.at(pos, i) = tokenEmbeddingTable.at(token, i) + posEmbeddingTable.at(pos, i);
        }
    }
    
    // 2. Przejście przez warstwy Transformera
    for(int l = 0; l < n_layers; l++) {
        Tensor residual = x;

        // -- Self Attention --
        x.applyRMSNorm(layers[l].rmsAttn);

        Tensor q = Tensor::matmul(x, layers[l].wQ);
        Tensor k = Tensor::matmul(x, layers[l].wK);
        Tensor v = Tensor::matmul(x, layers[l].wV);

        // Obliczanie atencji (q * k^T)
        Tensor scores = Tensor::matmul(q, k, true); // [T, T]
        
        float scale = 1.0f / std::sqrt((float)dim);
        for(int r = 0; r < T; r++) {
            float max_val = -1e9f;
            for(int c = 0; c < T; c++) {
                if (c > r) {
                    scores.at(r, c) = -1e9f; // Causal mask
                } else {
                    scores.at(r, c) *= scale;
                    if (scores.at(r, c) > max_val) max_val = scores.at(r, c);
                }
            }
            // Softmax per row
            float sum = 0.0f;
            for(int c = 0; c <= r; c++) {
                scores.at(r, c) = std::exp(scores.at(r, c) - max_val);
                sum += scores.at(r, c);
            }
            for(int c = 0; c <= r; c++) {
                scores.at(r, c) /= sum;
            }
            // Zmaskowane wartości muszą być jawnie równe 0.0f do matmul!
            for(int c = r + 1; c < T; c++) {
                scores.at(r, c) = 0.0f;
            }
        }

        Tensor attn_out = Tensor::matmul(scores, v);

        // Projekcja wyjściowa
        x = Tensor::matmul(attn_out, layers[l].wO);

        // Residual Connection
        for(int r=0; r<T; r++) {
            for(int i=0; i<dim; i++) x.at(r, i) += residual.at(r, i);
        }

        // -- Feed Forward --
        residual = x;
        x.applyRMSNorm(layers[l].rmsFFN);

        Tensor gate = Tensor::matmul(x, layers[l].wGate);
        gate.applySiLU();
        Tensor up = Tensor::matmul(x, layers[l].wUp);
        
        Tensor ffn_mid({T, hidden_dim});
        for(int r=0; r<T; r++) {
            for(int i=0; i<hidden_dim; i++) ffn_mid.at(r, i) = gate.at(r, i) * up.at(r, i);
        }

        x = Tensor::matmul(ffn_mid, layers[l].wDown);

        // Residual Connection
        for(int r=0; r<T; r++) {
            for(int i=0; i<dim; i++) x.at(r, i) += residual.at(r, i);
        }
    }
    
    // 3. Klasyfikator końcowy dla OSTATNIEGO tokena
    Tensor last_token({1, dim});
    for(int i=0; i<dim; i++) {
        last_token.at(0, i) = x.at(T - 1, i);
    }
    
    last_token.applyRMSNorm(outputRMSNorm);
    Tensor logits = Tensor::matmul(last_token, outputClassifier);

    return logits;
}

std::string NppAIEngine::generate(const std::string& prompt, int maxTokens) {
    if (dim == 0) return "Model nie jest zaladowany!";

    std::vector<int> tokens = tokenize(prompt);
    
    // Główna pętla autoregresyjna AI
    for(int i = 0; i < maxTokens; i++) {
        Tensor logits = forward(tokens);
        
        // Wybieramy token z najwyższym prawdopodobieństwem (Argmax)
        int nextToken = 0;
        float maxVal = -1e9f;
        for(int v=0; v<vocab_size; v++) {
            if (logits.at(0, v) > maxVal) {
                maxVal = logits.at(0, v);
                nextToken = v;
            }
        }
        
        tokens.push_back(nextToken);
        
        // Warunek stopu (zakładamy 0 jako EOS)
        if (nextToken == 0) break;
    }
    
    return detokenize(tokens);
}
