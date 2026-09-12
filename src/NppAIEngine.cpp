#include "NppAIEngine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h> // SIMD/AVX2 support
#define USE_AVX2
#endif
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <d3d11.h>
#include <d3dcompiler.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

// Struktura przechowująca stan globalny GPU
struct GPUContext {
  ID3D11Device *device = nullptr;
  ID3D11DeviceContext *context = nullptr;
  ID3D11ComputeShader *matmulShader = nullptr;
  bool initialized = false;
  bool failed = false;

  ~GPUContext() {
    if (matmulShader)
      matmulShader->Release();
    if (context)
      context->Release();
    if (device)
      device->Release();
  }
};

static GPUContext g_gpu;

// Kod źródłowy HLSL (Compute Shader) do mnożenia macierzy
const char *hlsl_matmul = R"(
cbuffer Dimensions : register(b0) {
    uint a_rows;
    uint a_cols;
    uint b_effective_cols;
    uint transposeB;
};

StructuredBuffer<float> bufA : register(t0);
StructuredBuffer<float> bufB : register(t1);
RWStructuredBuffer<float> bufC : register(u0);

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint row = DTid.y;
    uint col = DTid.x;

    if (row < a_rows && col < b_effective_cols) {
        float sum = 0.0f;
        for (uint k = 0; k < a_cols; k++) {
            float b_val = transposeB ? bufB[col * a_cols + k] : bufB[k * b_effective_cols + col];
            sum += bufA[row * a_cols + k] * b_val;
        }
        bufC[row * b_effective_cols + col] = sum;
    }
}
)";

bool initGPU() {
  if (g_gpu.initialized)
    return true;
  if (g_gpu.failed)
    return false;

  D3D_FEATURE_LEVEL featureLevel;
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                 nullptr, 0, D3D11_SDK_VERSION, &g_gpu.device,
                                 &featureLevel, &g_gpu.context);
  if (FAILED(hr)) {
    std::cerr << "GPU: Brak sprzetu D3D11. Przelaczam na CPU (OpenMP).\n";
    g_gpu.failed = true;
    return false;
  }

  ID3DBlob *shaderBlob = nullptr;
  ID3DBlob *errorBlob = nullptr;
  hr = D3DCompile(hlsl_matmul, strlen(hlsl_matmul), nullptr, nullptr, nullptr,
                  "main", "cs_5_0", 0, 0, &shaderBlob, &errorBlob);
  if (FAILED(hr)) {
    if (errorBlob) {
      std::cerr << "GPU Shader Error: " << (char *)errorBlob->GetBufferPointer()
                << "\n";
      errorBlob->Release();
    }
    g_gpu.failed = true;
    return false;
  }

  hr = g_gpu.device->CreateComputeShader(shaderBlob->GetBufferPointer(),
                                         shaderBlob->GetBufferSize(), nullptr,
                                         &g_gpu.matmulShader);
  shaderBlob->Release();
  if (FAILED(hr)) {
    g_gpu.failed = true;
    return false;
  }

  g_gpu.initialized = true;
  std::cout << "GPU: DirectCompute (DirectX 11) zainicjowane pomyslnie! Karta "
               "graficzna gotowa.\n";
  return true;
}

bool matmul_gpu(const Tensor &a, const Tensor &b, Tensor &result,
                bool transposeB) {
  if (!initGPU())
    return false;

  int a_rows = a.shape[0];
  int a_cols = a.shape[1];
  int b_effective_cols = transposeB ? b.shape[0] : b.shape[1];

  if (a_rows == 0 || a_cols == 0 || b_effective_cols == 0)
    return false;

  // Tworzenie buforów wejściowych (A i B)
  D3D11_BUFFER_DESC descA = {};
  descA.Usage = D3D11_USAGE_DEFAULT;
  descA.ByteWidth = (UINT)(a.data.size() * sizeof(float));
  descA.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  descA.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
  descA.StructureByteStride = sizeof(float);
  D3D11_SUBRESOURCE_DATA initA = {};
  initA.pSysMem = a.data.data();
  ID3D11Buffer *pBufA = nullptr;
  if (FAILED(g_gpu.device->CreateBuffer(&descA, &initA, &pBufA)))
    return false;

  D3D11_BUFFER_DESC descB = descA;
  D3D11_SUBRESOURCE_DATA initB = {};
  std::vector<float> tempB;
  if (!b.data_q8.empty()) {
    descB.ByteWidth = (UINT)(b.data_q8.size() * sizeof(float));
    tempB.resize(b.data_q8.size());
    for (size_t i = 0; i < b.data_q8.size(); i++) {
      tempB[i] = b.data_q8[i] * b.scale_q8;
    }
    initB.pSysMem = tempB.data();
  } else {
    descB.ByteWidth = (UINT)(b.data.size() * sizeof(float));
    initB.pSysMem = b.data.data();
  }
  ID3D11Buffer *pBufB = nullptr;
  if (FAILED(g_gpu.device->CreateBuffer(&descB, &initB, &pBufB))) {
    pBufA->Release();
    return false;
  }

  // Tworzenie bufora wyjściowego (C)
  D3D11_BUFFER_DESC descC = descA;
  descC.ByteWidth = (UINT)(result.data.size() * sizeof(float));
  descC.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
  ID3D11Buffer *pBufC = nullptr;
  if (FAILED(g_gpu.device->CreateBuffer(&descC, nullptr, &pBufC))) {
    pBufA->Release();
    pBufB->Release();
    return false;
  }

  // Tworzenie widoków (Views) do Shadera
  D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
  srvDesc.Format = DXGI_FORMAT_UNKNOWN;
  srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
  srvDesc.Buffer.FirstElement = 0;
  srvDesc.Buffer.NumElements = (UINT)a.data.size();
  ID3D11ShaderResourceView *pSrvA = nullptr;
  g_gpu.device->CreateShaderResourceView(pBufA, &srvDesc, &pSrvA);

  srvDesc.Buffer.NumElements =
      (UINT)(b.data_q8.empty() ? b.data.size() : b.data_q8.size());
  ID3D11ShaderResourceView *pSrvB = nullptr;
  g_gpu.device->CreateShaderResourceView(pBufB, &srvDesc, &pSrvB);

  D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.Format = DXGI_FORMAT_UNKNOWN;
  uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
  uavDesc.Buffer.FirstElement = 0;
  uavDesc.Buffer.NumElements = (UINT)result.data.size();
  ID3D11UnorderedAccessView *pUavC = nullptr;
  g_gpu.device->CreateUnorderedAccessView(pBufC, &uavDesc, &pUavC);

  // Przekazanie wymiarów do Shadera (Constant Buffer)
  struct Constants {
    uint32_t ar, ac, bc, tb;
  };
  Constants consts = {(uint32_t)a_rows, (uint32_t)a_cols,
                      (uint32_t)b_effective_cols,
                      (uint32_t)(transposeB ? 1 : 0)};
  D3D11_BUFFER_DESC cbDesc = {};
  cbDesc.Usage = D3D11_USAGE_DEFAULT;
  cbDesc.ByteWidth = sizeof(Constants);
  cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  D3D11_SUBRESOURCE_DATA cbInit = {};
  cbInit.pSysMem = &consts;
  ID3D11Buffer *pCB = nullptr;
  g_gpu.device->CreateBuffer(&cbDesc, &cbInit, &pCB);

  // Wywołanie Shadera Obliczeniowego (Dispatch)
  g_gpu.context->CSSetShader(g_gpu.matmulShader, nullptr, 0);
  ID3D11ShaderResourceView *srvs[] = {pSrvA, pSrvB};
  g_gpu.context->CSSetShaderResources(0, 2, srvs);
  g_gpu.context->CSSetUnorderedAccessViews(0, 1, &pUavC, nullptr);
  g_gpu.context->CSSetConstantBuffers(0, 1, &pCB);

  uint32_t dispatchX = (b_effective_cols + 15) / 16;
  uint32_t dispatchY = (a_rows + 15) / 16;
  g_gpu.context->Dispatch(dispatchX, dispatchY, 1);

  // Czyszczenie przypisań, by zwolnić zasoby
  ID3D11ShaderResourceView *nullSRV[] = {nullptr, nullptr};
  g_gpu.context->CSSetShaderResources(0, 2, nullSRV);
  ID3D11UnorderedAccessView *nullUAV[] = {nullptr};
  g_gpu.context->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

  // Odczytanie wyników z VRAM z powrotem do RAM
  D3D11_BUFFER_DESC readDesc = {};
  readDesc.Usage = D3D11_USAGE_STAGING;
  readDesc.ByteWidth = (UINT)(result.data.size() * sizeof(float));
  readDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  ID3D11Buffer *pReadBuf = nullptr;
  g_gpu.device->CreateBuffer(&readDesc, nullptr, &pReadBuf);

  g_gpu.context->CopyResource(pReadBuf, pBufC);

  D3D11_MAPPED_SUBRESOURCE mapped;
  if (SUCCEEDED(g_gpu.context->Map(pReadBuf, 0, D3D11_MAP_READ, 0, &mapped))) {
    memcpy(result.data.data(), mapped.pData,
           result.data.size() * sizeof(float));
    g_gpu.context->Unmap(pReadBuf, 0);
  }

  // Sprzątanie pamięci karty graficznej
  pReadBuf->Release();
  pCB->Release();
  if (pUavC)
    pUavC->Release();
  if (pSrvB)
    pSrvB->Release();
  if (pSrvA)
    pSrvA->Release();
  pBufC->Release();
  pBufB->Release();
  pBufA->Release();

  return true;
}

// --- TENSOR IMPLEMENTATION ---
Tensor::Tensor(std::vector<int> s) : shape(s) {
  int size = 1;
  for (int d : shape)
    size *= d;
  data.resize(size, 0.0f);
}

float &Tensor::at(int i) { return data[i]; }
float &Tensor::at(int r, int c) { return data[r * shape[1] + c]; }

// Mnożenie macierzy zoptymalizowane za pomocą OpenMP i GPU (DirectX 11)
Tensor Tensor::matmul(const Tensor &a, const Tensor &b, bool transposeB) {
  int a_rows = a.shape[0];
  int a_cols = a.shape[1];
  int b_rows = transposeB ? b.shape[1] : b.shape[0];
  int b_cols = transposeB ? b.shape[0] : b.shape[1];

  Tensor result({a_rows, b_cols});

  // HYBRYDOWY SILNIK:
  // Jeśli macierz jest duża (np. faza przetwarzania całego promptu), używamy
  // GPU. Dla małych macierzy (np. generowanie pojedynczego tokena) używamy CPU
  // OpenMP, ponieważ kopiowanie danych z RAM do VRAM dla małych ilości danych
  // zajęłoby więcej czasu niż samo liczenie na CPU.
  if (a_rows >= 16) {
    if (matmul_gpu(a, b, result, transposeB)) {
      return result; // GPU policzyło i zwróciło wynik
    }
  }

// Fallback: CPU (OpenMP + AVX2/Scalar)
#pragma omp parallel for
  for (int i = 0; i < a_rows; i++) {
    for (int j = 0; j < b_cols; j++) {
      float sum = 0.0f;
      if (transposeB) {
        if (!b.data_q8.empty()) {
#ifdef USE_AVX2
          int k = 0;
          __m256 sum_vec = _mm256_setzero_ps();
          __m256 scale_vec = _mm256_set1_ps(b.scale_q8);
          for (; k <= a_cols - 8; k += 8) {
            __m256 va = _mm256_loadu_ps(&a.data[i * a.shape[1] + k]);
            __m128i vb_int8 =
                _mm_loadl_epi64((__m128i *)&b.data_q8[j * b.shape[1] + k]);
            __m256i vb_int32 = _mm256_cvtepi8_epi32(vb_int8);
            __m256 vb_float = _mm256_cvtepi32_ps(vb_int32);
            vb_float = _mm256_mul_ps(vb_float, scale_vec);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb_float));
          }
          float tmp[8];
          _mm256_storeu_ps(tmp, sum_vec);
          for (int m = 0; m < 8; m++)
            sum += tmp[m];
          for (; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] *
                   (b.data_q8[j * b.shape[1] + k] * b.scale_q8);
          }
#else
          for (int k = 0; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] *
                   (b.data_q8[j * b.shape[1] + k] * b.scale_q8);
          }
#endif
        } else {
#ifdef USE_AVX2
          int k = 0;
          __m256 sum_vec = _mm256_setzero_ps();
          for (; k <= a_cols - 8; k += 8) {
            __m256 va = _mm256_loadu_ps(&a.data[i * a.shape[1] + k]);
            __m256 vb = _mm256_loadu_ps(&b.data[j * b.shape[1] + k]);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
          }
          float tmp[8];
          _mm256_storeu_ps(tmp, sum_vec);
          for (int m = 0; m < 8; m++)
            sum += tmp[m];
          for (; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] * b.data[j * b.shape[1] + k];
          }
#else
          for (int k = 0; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] * b.data[j * b.shape[1] + k];
          }
#endif
        }
      } else {
        if (!b.data_q8.empty()) {
          for (int k = 0; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] *
                   (b.data_q8[k * b.shape[1] + j] * b.scale_q8);
          }
        } else {
          for (int k = 0; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
          }
        }
      }
      result.data[i * b_cols + j] = sum;
    }
  }
  return result;
}

void Tensor::applySiLU() {
  for (auto &val : data) {
    val = val / (1.0f + std::exp(-val)); // x * sigmoid(x)
  }
}

void Tensor::applyRMSNorm(const Tensor &weight) {
  int rows = shape.size() > 1 ? shape[0] : 1;
  int cols = shape.size() > 1 ? shape[1] : shape[0];

#pragma omp parallel for
  for (int r = 0; r < rows; r++) {
    float ss = 0.0f;
    int c = 0;

#ifdef USE_AVX2
    // Faza 1: Suma kwadratów z AVX2
    __m256 sum_vec = _mm256_setzero_ps();
    for (; c <= cols - 8; c += 8) {
      __m256 val = _mm256_loadu_ps(&data[r * cols + c]);
      sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(val, val));
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sum_vec);
    for (int i = 0; i < 8; i++)
      ss += tmp[i];
#endif

    // Reszta / Scalar
    for (; c < cols; c++) {
      float val = data[r * cols + c];
      ss += val * val;
    }

    ss /= cols;
    ss += 1e-5f; // epsilon
    ss = 1.0f / std::sqrt(ss);

    c = 0;
#ifdef USE_AVX2
    // Faza 2: Normalizacja z AVX2
    __m256 ss_vec = _mm256_set1_ps(ss);
    for (; c <= cols - 8; c += 8) {
      __m256 val = _mm256_loadu_ps(&data[r * cols + c]);
      __m256 w = _mm256_loadu_ps(&weight.data[c]);
      __m256 res = _mm256_mul_ps(_mm256_mul_ps(val, ss_vec), w);
      _mm256_storeu_ps(&data[r * cols + c], res);
    }
#endif

    // Reszta / Scalar
    for (; c < cols; c++) {
      data[r * cols + c] = (data[r * cols + c] * ss) * weight.data[c];
    }
  }
}

float Tensor::get(int i) const {
  if (!data_q8.empty()) {
    return data_q8[i] * scale_q8;
  }
  return data[i];
}

float Tensor::get(int r, int c) const {
  if (!data_q8.empty()) {
    return data_q8[r * shape[1] + c] * scale_q8;
  }
  return data[r * shape[1] + c];
}

void Tensor::readFromFile(std::ifstream &file, bool quantize) {
  file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));

  if (quantize) {
    float max_abs = 0.0f;
    for (float val : data) {
      if (std::abs(val) > max_abs)
        max_abs = std::abs(val);
    }
    scale_q8 = max_abs / 127.0f;
    if (scale_q8 == 0.0f)
      scale_q8 = 1e-9f;

    data_q8.resize(data.size());
    for (size_t i = 0; i < data.size(); i++) {
      data_q8[i] = static_cast<int8_t>(std::round(data[i] / scale_q8));
    }

    // Zwalniamy oryginalne dane zmiennoprzecinkowe dla oszczędności RAM
    data.clear();
    data.shrink_to_fit();
  }
}

// --- ENGINE IMPLEMENTATION ---
NppAIEngine::NppAIEngine() {}

NppAIEngine::~NppAIEngine() {}

bool NppAIEngine::loadModel(const std::string &modelPath) {
  std::lock_guard<std::mutex> lock(engineMutex);
  std::ifstream file(modelPath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Nie udalo sie otworzyc pliku modelu: " << modelPath
              << std::endl;
    return false;
  }

  // Wczytywanie nagłówka (konfiguracji)
  file.read(reinterpret_cast<char *>(&dim), sizeof(int));
  file.read(reinterpret_cast<char *>(&hidden_dim), sizeof(int));
  file.read(reinterpret_cast<char *>(&n_layers), sizeof(int));
  file.read(reinterpret_cast<char *>(&max_seq_len), sizeof(int));
  file.read(reinterpret_cast<char *>(&vocab_size), sizeof(int));

  // Inicjalizacja i wczytywanie wag
  tokenEmbeddingTable = Tensor({vocab_size, dim});
  tokenEmbeddingTable.readFromFile(file, false); // Embeddings usually stay FP32

  posEmbeddingTable = Tensor({max_seq_len, dim});
  posEmbeddingTable.readFromFile(file, false);

  layers.clear();
  for (int i = 0; i < n_layers; i++) {
    TransformerLayer layer;
    layer.rmsAttn = Tensor({dim});
    layer.rmsAttn.readFromFile(file, false); // RMSNorm is small, FP32
    layer.wQ = Tensor({dim, dim});
    layer.wQ.readFromFile(file, true); // Quantize
    layer.wK = Tensor({dim, dim});
    layer.wK.readFromFile(file, true);
    layer.wV = Tensor({dim, dim});
    layer.wV.readFromFile(file, true);
    layer.wO = Tensor({dim, dim});
    layer.wO.readFromFile(file, true);

    layer.rmsFFN = Tensor({dim});
    layer.rmsFFN.readFromFile(file, false);
    layer.wGate = Tensor({dim, hidden_dim});
    layer.wGate.readFromFile(file, true);
    layer.wUp = Tensor({dim, hidden_dim});
    layer.wUp.readFromFile(file, true);
    layer.wDown = Tensor({hidden_dim, dim});
    layer.wDown.readFromFile(file, true);

    layers.push_back(layer);
  }

  outputRMSNorm = Tensor({dim});
  outputRMSNorm.readFromFile(file, false);

  outputClassifier = Tensor({dim, vocab_size});
  outputClassifier.readFromFile(file, true); // Quantize output classifier

  file.close();

  // Wczytanie BPE tokenizera
  std::string bpePath =
      modelPath.substr(0, modelPath.find_last_of("/\\")) + "\\bpe_merges.txt";
  loadBPETokenizer(bpePath);

  return true;
}

bool NppAIEngine::loadBPETokenizer(const std::string &path) {
  bpe_merges.clear();
  bpe_vocab.clear();
  for (int i = 0; i < 256; i++) {
    bpe_vocab[i] = std::string(1, (char)i);
  }

  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Nie udalo sie wczytac BPE Tokenizera: " << path
              << ". Uzywany tryb bajtowy.\n";
    return false;
  }

  int p0, p1, idx;
  while (file >> p0 >> p1 >> idx) {
    bpe_merges[{p0, p1}] = idx;
    bpe_vocab[idx] = bpe_vocab[p0] + bpe_vocab[p1];
  }
  return true;
}

std::vector<int> NppAIEngine::tokenize(const std::string &text) {
  std::vector<int> ids;
  for (char c : text)
    ids.push_back((unsigned char)c);

  if (bpe_merges.empty())
    return ids; // Fallback do bajtów

  while (ids.size() >= 2) {
    int best_idx = -1;
    std::pair<int, int> best_pair;
    int min_rank = 1000000000;

    for (size_t i = 0; i < ids.size() - 1; i++) {
      std::pair<int, int> pair = {ids[i], ids[i + 1]};
      if (bpe_merges.count(pair)) {
        if (bpe_merges[pair] < min_rank) {
          min_rank = bpe_merges[pair];
          best_pair = pair;
        }
      }
    }

    if (min_rank == 1000000000)
      break;

    std::vector<int> new_ids;
    for (size_t i = 0; i < ids.size(); i++) {
      if (i < ids.size() - 1 && ids[i] == best_pair.first &&
          ids[i + 1] == best_pair.second) {
        new_ids.push_back(min_rank);
        i++;
      } else {
        new_ids.push_back(ids[i]);
      }
    }
    ids = new_ids;
  }
  return ids;
}

std::string NppAIEngine::detokenize(const std::vector<int> &tokens) {
  std::string text;
  for (int t : tokens) {
    if (bpe_vocab.count(t)) {
      text += bpe_vocab[t];
    } else if (t >= 0 && t < 256) {
      text += (char)(unsigned char)t;
    }
  }
  return text;
}

Tensor NppAIEngine::forward(const std::vector<int> &inputTokens) {
  int seq_len = (int)inputTokens.size();
  if (seq_len == 0 || dim == 0)
    return Tensor({1, vocab_size});

  // Zabezpieczenie przed przekroczeniem kontekstu
  int T = seq_len < max_seq_len ? seq_len : max_seq_len;

  // 1. Embedding dla całej sekwencji T
  Tensor x({T, dim});
  for (int pos = 0; pos < T; pos++) {
    int token = inputTokens[seq_len - T + pos]; // Bierzemy T ostatnich tokenów
    if (token >= vocab_size || token < 0) {
      token = 0; // clamp to 0 to prevent segfault if vocab_size mismatch
    }
    for (int i = 0; i < dim; i++) {
      x.at(pos, i) =
          tokenEmbeddingTable.at(token, i) + posEmbeddingTable.at(pos, i);
    }
  }

  // 2. Przejście przez warstwy Transformera
  for (int l = 0; l < n_layers; l++) {
    Tensor residual = x;

    // -- Self Attention --
    x.applyRMSNorm(layers[l].rmsAttn);

    Tensor q = Tensor::matmul(x, layers[l].wQ);
    Tensor k = Tensor::matmul(x, layers[l].wK);
    Tensor v = Tensor::matmul(x, layers[l].wV);

    // Obliczanie atencji (q * k^T)
    Tensor scores = Tensor::matmul(q, k, true); // [T, T]

    float scale = 1.0f / std::sqrt((float)dim);
#pragma omp parallel for
    for (int r = 0; r < T; r++) {
      float max_val = -1e9f;
      for (int c = 0; c < T; c++) {
        if (c > r) {
          scores.at(r, c) = -1e9f; // Causal mask
        } else {
          scores.at(r, c) *= scale;
          if (scores.at(r, c) > max_val)
            max_val = scores.at(r, c);
        }
      }
      // Softmax per row
      float sum = 0.0f;
      for (int c = 0; c <= r; c++) {
        scores.at(r, c) = std::exp(scores.at(r, c) - max_val);
        sum += scores.at(r, c);
      }
      for (int c = 0; c <= r; c++) {
        scores.at(r, c) /= sum;
      }
      // Zmaskowane wartości muszą być jawnie równe 0.0f do matmul!
      for (int c = r + 1; c < T; c++) {
        scores.at(r, c) = 0.0f;
      }
    }

    Tensor attn_out = Tensor::matmul(scores, v);

    // Projekcja wyjściowa
    x = Tensor::matmul(attn_out, layers[l].wO);

// Residual Connection
#pragma omp parallel for
    for (int r = 0; r < T; r++) {
      for (int i = 0; i < dim; i++)
        x.data[r * dim + i] += residual.data[r * dim + i];
    }

    // -- Feed Forward --
    residual = x;
    x.applyRMSNorm(layers[l].rmsFFN);

    Tensor gate = Tensor::matmul(x, layers[l].wGate);
    gate.applySiLU();
    Tensor up = Tensor::matmul(x, layers[l].wUp);

    Tensor ffn_mid({T, hidden_dim});
#pragma omp parallel for
    for (int r = 0; r < T; r++) {
      for (int i = 0; i < hidden_dim; i++)
        ffn_mid.data[r * hidden_dim + i] =
            gate.data[r * hidden_dim + i] * up.data[r * hidden_dim + i];
    }

    x = Tensor::matmul(ffn_mid, layers[l].wDown);

// Residual Connection
#pragma omp parallel for
    for (int r = 0; r < T; r++) {
      for (int i = 0; i < dim; i++)
        x.data[r * dim + i] += residual.data[r * dim + i];
    }
  }

  // 3. Klasyfikator końcowy dla OSTATNIEGO tokena
  Tensor last_token({1, dim});
  for (int i = 0; i < dim; i++) {
    last_token.at(0, i) = x.at(T - 1, i);
  }

  last_token.applyRMSNorm(outputRMSNorm);
  Tensor logits = Tensor::matmul(last_token, outputClassifier);

  return logits;
}

std::string NppAIEngine::generate(const std::string &prompt, int maxTokens,
                                  std::function<void(char, bool)> onToken,
                                  std::function<void(int)> onRemove) {
  std::lock_guard<std::mutex> lock(engineMutex);
  if (dim == 0)
    return "Model nie jest zaladowany!";

  cancelRequested = false;
  std::vector<int> tokens = tokenize(prompt);
  std::string current_output = prompt;
  bool is_thinking = false;

  // Główna pętla autoregresyjna AI
  for (int i = 0; i < maxTokens; i++) {
    if (cancelRequested) {
      std::cout << "\n[Generowanie przerwane przez uzytkownika]" << std::endl;
      break;
    }

    Tensor logits = forward(tokens);

    // Zabezpieczenie przed NaN (wybuchami w matematyce Tensorowej)
    bool has_nan = false;
    for (int v = 0; v < vocab_size; v++) {
      if (std::isnan(logits.at(0, v))) {
        has_nan = true;
        break;
      }
    }

    int nextToken = 0;
    if (has_nan) {
      nextToken = 0; // Fallback na bezpieczny token
    } else {
      // Repetition Penalty - obniżamy szansę na znaki, które wystąpiły niedawno
      // w kontekście
      float repetition_penalty = 1.3f; // Zwiększone z 1.2 na 1.3
      for (int t : tokens) {
        if (logits.at(0, t) > 0) {
          logits.data[t] /= repetition_penalty;
        } else {
          logits.data[t] *= repetition_penalty;
        }
      }

      // TOP-K Sampling (Rozwiązanie problemu "pustych spacji" i krzaczków)
      // Zamiast brać absolutnie największą wartość (Greedy) lub losować ze
      // wszystkich, ograniczamy wybór tylko do K najbardziej prawdopodobnych
      // liter.
      int K = 3; // Zmniejszamy K z 5 na 3, aby ograniczyć zniekształcenia
                 // (halucynacje)
      std::vector<std::pair<float, int>> top_logits;
      for (int v = 0; v < vocab_size; ++v) {
        top_logits.push_back({logits.at(0, v), v});
      }

      // Sortowanie malejąco
      std::sort(
          top_logits.begin(), top_logits.end(),
          [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
            return a.first > b.first;
          });

      // Temperatura decyzyjna
      float temperature = 0.35f; // Zmniejszamy z 0.5 na 0.35, aby model był
                                 // "pewniejszy" i mniej zgadywał
      std::vector<float> probs(K, 0.0f);
      float sum_probs = 0.0f;

      // Wyciągnięcie prawdopodobieństw tylko dla Top-K znaków
      for (int j = 0; j < K; ++j) {
        probs[j] = std::exp(top_logits[j].first / temperature);
        sum_probs += probs[j];
      }

      // Rzutowanie losowe (Weighted Random) z Top-K
      float r = (float)rand() / (float)RAND_MAX;
      float cumulative = 0.0f;
      bool selected = false;
      for (int j = 0; j < K; ++j) {
        cumulative += probs[j] / sum_probs;
        if (r <= cumulative) {
          nextToken = top_logits[j].second;
          selected = true;
          break;
        }
      }
      if (!selected) {
        nextToken = top_logits[0].second; // Fallback na najlepszą literę
      }

      // Zabezpieczenie przed niekontrolowanymi znakami kontrolnymi ASCII
      // (czasami model próbuje wypluć null-bajty co w edytorze wygląda jak
      // puste bloki)
      if (nextToken < 32 && nextToken != '\n' && nextToken != '\r' &&
          nextToken != '\t') {
        nextToken = ' '; // Bezpieczny zamiennik
      }
    }

    tokens.push_back(nextToken);

    // Warunek stopu (zakładamy 0 jako EOS)
    if (nextToken == 0)
      break;

    // Zatrzymujemy generowanie od razu, jeśli model próbuje rozpocząć nową
    // "rozmowę"
    char c = (char)(unsigned char)nextToken;

    // Buforowanie, by wykryć "[USER]" (model halucynuje, że on sam jest
    // użytkownikiem)
    current_output += c;
    if (current_output.find("[USER]") != std::string::npos ||
        current_output.find("[SYSTEM]") != std::string::npos) {
      // AI zwariowało i weszło w pętle. Usuwamy ostatnie 6 znaków ("[USER]") z
      // edytora
      if (onRemove) {
        onRemove(6); // Backspace 6 razy
      }
      break;
    }
    if (nextToken >= 0 && nextToken < 256) {
      // char c = (char)(unsigned char)nextToken; // Zmienna "c" już jest
      // zadeklarowana wyżej! current_output += c; // Buforowanie już jest
      // robione wyżej!

      // Sprawdzamy czy to nie początek myślenia
      if (!is_thinking && current_output.length() >= 7 &&
          current_output.substr(current_output.length() - 7) == "<THINK>") {
        is_thinking = true;
        // Usuń "<THINK>" z edytora
        if (onRemove)
          onRemove(7);
        continue; // nie wywołujemy onToken dla tego znaku
      }

      // Sprawdzamy czy to nie koniec myślenia
      else if (is_thinking && current_output.length() >= 8 &&
               current_output.substr(current_output.length() - 8) ==
                   "</THINK>") {
        is_thinking = false;
        continue;
      }

      // Wypisujemy znak na ekran w czasie rzeczywistym
      if (onToken) {
        onToken(c, is_thinking);
      }

      // Sprawdzamy czy na końcu wygenerowanego tekstu nie pojawił się tag
      // nowego promptu
      if (current_output.length() >= 7 &&
          current_output.substr(current_output.length() - 7) == "[USER]:") {

        // Callback do usunięcia tagu "[USER]:" z edytora
        if (onRemove)
          onRemove(7);

        // Obcinamy "[USER]:" z końcowej listy tokenów i wychodzimy
        for (int j = 0; j < 7; j++)
          tokens.pop_back();
        break;
      }
    }
  }

  std::cout << std::endl;
  return detokenize(tokens);
}