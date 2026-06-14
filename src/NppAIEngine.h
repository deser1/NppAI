// NppAIEngine.h
#pragma once
#include <vector>
#include <string>
#include <memory>

// Reprezentacja Tensora (macierzy wielowymiarowej) w naszym własnym silniku
class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() = default;
    Tensor(std::vector<int> s);
    
    float& at(int i);
    float& at(int r, int c);
    
    // Podstawowa operacja dla sieci neuronowych: y = x * W
    static Tensor matmul(const Tensor& a, const Tensor& b, bool transposeB = false);
    
    // Funkcja aktywacji (np. SiLU lub GELU stosowane w nowoczesnych modelach)
    void applySiLU();

    // Normalizacja RMSNorm
    void applyRMSNorm(const Tensor& weight);

    // Wczytanie surowych danych z pliku
    void readFromFile(std::ifstream& file);
};

// Klasa reprezentująca wagę pojedynczej warstwy Transformera
struct TransformerLayer {
    // Wagi dla Self-Attention
    Tensor wQ, wK, wV, wO;
    // Wagi dla Feed-Forward Network
    Tensor wGate, wDown, wUp;
    // Layer Normalization
    Tensor rmsAttn, rmsFFN;
};

// Główny silnik inferencji modelu NppAI
class NppAIEngine {
public:
    NppAIEngine();
    ~NppAIEngine();

    // Wczytuje nasz autorski, binarny plik z wagami modelu
    bool loadModel(const std::string& modelPath);

    // Generuje kod na podstawie promptu
    std::string generate(const std::string& prompt, int maxTokens = 128);

private:
    // Parametry modelu (np. wielkość osadzeń, liczba głów)
    int dim = 0;
    int hidden_dim = 0;
    int n_layers = 0;
    int max_seq_len = 0;
    int vocab_size = 0;

    // Wagi całego modelu
    Tensor tokenEmbeddingTable;
    Tensor posEmbeddingTable;
    std::vector<TransformerLayer> layers;
    Tensor outputRMSNorm;
    Tensor outputClassifier;

    // Wewnętrzne operacje Transformera
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& tokens);
    Tensor forward(const std::vector<int>& inputTokens);
};
