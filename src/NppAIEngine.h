// NppAIEngine.h
#pragma once
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <map>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int8_t> data_q8;
    float scale_q8 = 0.0f;
    std::vector<int> shape;

    Tensor() = default;
    Tensor(std::vector<int> s);

    float& at(int i);
    float& at(int r, int c);

    float get(int i) const;
    float get(int r, int c) const;

    static Tensor matmul(const Tensor& a, const Tensor& b, bool transposeB = false);
    void applySiLU();
    void applyRMSNorm(const Tensor& weight);

    bool readFromFile(std::ifstream& file, bool quantize = false);

#ifdef NPPAI_TESTING
    static void setSimdOverrideForTesting(int mode);
    static bool simdAvailableForTesting();
    static void setGpuOverrideForTesting(int mode);
    static bool gpuAvailableForTesting();
#endif
};

struct MoERoute {
    int expertIndex = 0;
    float weight = 0.0f;
    bool usedFallback = false;
};

class MoERouter {
public:
    // Selects the k highest finite logits. Ties are resolved by the lower
    // expert index, and returned weights are a softmax over selected logits.
    static std::vector<MoERoute> topK(const std::vector<float>& logits, int k);

    // Routes one token while respecting per-expert capacity. Preferred
    // experts are considered in deterministic logit/index order. If fewer
    // than k preferred experts have capacity, remaining experts are used as
    // deterministic fallbacks. Throws when total remaining capacity is
    // insufficient to route the token.
    static std::vector<MoERoute> topKWithCapacity(
        const std::vector<float>& logits,
        int k,
        std::vector<int>& expertLoads,
        int capacityPerExpert);
};

struct TransformerLayer {
    Tensor wQ, wK, wV, wO;
    Tensor wGate, wDown, wUp;
    Tensor rmsAttn, rmsFFN;
};

class NppAIEngine {
public:
    NppAIEngine();
    ~NppAIEngine();

    bool loadModel(const std::string& modelPath);
    std::string generate(
        const std::string& prompt,
        int maxTokens = 512,
        std::function<void(char, bool)> onToken = nullptr,
        std::function<void(int)> onRemove = nullptr);

    void stopGeneration() { cancelRequested = true; }

private:
    std::atomic<bool> cancelRequested{false};
    std::mutex engineMutex;

    int dim = 0;
    int hidden_dim = 0;
    int n_layers = 0;
    int max_seq_len = 0;
    int vocab_size = 0;

    Tensor tokenEmbeddingTable;
    Tensor posEmbeddingTable;
    std::vector<TransformerLayer> layers;
    Tensor outputRMSNorm;
    Tensor outputClassifier;

    std::map<std::pair<int, int>, int> bpe_merges;
    std::map<std::pair<int, int>, size_t> bpe_merge_ranks;
    std::map<int, std::string> bpe_vocab;
    bool loadBPETokenizer(const std::string& path);

    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& tokens);
    Tensor forward(const std::vector<int>& inputTokens);

    friend class NppAITest;
};
