#include "NppAIEngine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <array>
#include <cstring>
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#define NPPAI_X86_SIMD 1
#ifdef _MSC_VER
#include <intrin.h>
#endif

static bool cpuSupportsAVX2FMA() {
#ifdef _MSC_VER
  int cpuInfo[4] = {};
  __cpuid(cpuInfo, 0);
  if (cpuInfo[0] < 7)
    return false;

  __cpuid(cpuInfo, 1);
  const bool osxsave = (cpuInfo[2] & (1 << 27)) != 0;
  const bool avx = (cpuInfo[2] & (1 << 28)) != 0;
  const bool fma = (cpuInfo[2] & (1 << 12)) != 0;
  if (!osxsave || !avx || !fma)
    return false;

  // XMM (bit 1) and YMM (bit 2) state must both be managed by the OS.
  if ((_xgetbv(0) & 0x6) != 0x6)
    return false;

  __cpuidex(cpuInfo, 7, 0);
  return (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
#else
  // The current native Windows build uses MSVC. Keep non-MSVC x86 builds on
  // the scalar path until equivalent CPUID/XGETBV handling is implemented.
  return false;
#endif
}

static bool detectedAVX2FMA() {
  static const bool supported = cpuSupportsAVX2FMA();
  return supported;
}

#ifdef NPPAI_TESTING
static int g_simdOverride = -1; // -1 = auto, 0 = scalar, 1 = SIMD when supported
#endif

static bool useAVX2FMA() {
#ifdef NPPAI_TESTING
  if (g_simdOverride == 0)
    return false;
  if (g_simdOverride == 1)
    return detectedAVX2FMA();
#endif
  return detectedAVX2FMA();
}
#else
static bool useAVX2FMA() { return false; }
#endif
#include <iostream>
#include <limits>
#include <stdexcept>

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

#ifdef NPPAI_TESTING
static int g_gpuOverride = -1; // -1 = auto, 0 = CPU only, 1 = GPU when available
#endif

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

  srvDesc.Buffer.NumElements = (UINT)(b.data_q8.empty() ? b.data.size() : b.data_q8.size());
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
#ifdef NPPAI_TESTING
void Tensor::setSimdOverrideForTesting(int mode) {
  g_simdOverride = mode;
}

bool Tensor::simdAvailableForTesting() {
  return detectedAVX2FMA();
}

void Tensor::setGpuOverrideForTesting(int mode) {
  g_gpuOverride = mode;
}

bool Tensor::gpuAvailableForTesting() {
  return initGPU();
}
#endif

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
  if (a.shape.size() != 2 || b.shape.size() != 2)
    throw std::invalid_argument("Tensor::matmul requires rank-2 tensors");
  if (a.shape[0] < 0 || a.shape[1] < 0 || b.shape[0] < 0 || b.shape[1] < 0)
    throw std::invalid_argument("Tensor::matmul requires non-negative dimensions");

  int a_rows = a.shape[0];
  int a_cols = a.shape[1];
  int b_rows = transposeB ? b.shape[1] : b.shape[0];
  int b_cols = transposeB ? b.shape[0] : b.shape[1];
  if (a_cols != b_rows)
    throw std::invalid_argument("Tensor::matmul dimension mismatch");

  const size_t aExpected = static_cast<size_t>(a_rows) * static_cast<size_t>(a_cols);
  const size_t bExpected = static_cast<size_t>(b.shape[0]) * static_cast<size_t>(b.shape[1]);
  if (a.data.size() != aExpected)
    throw std::invalid_argument("Tensor::matmul invalid left tensor storage");
  if ((!b.data_q8.empty() && b.data_q8.size() != bExpected) ||
      (b.data_q8.empty() && b.data.size() != bExpected))
    throw std::invalid_argument("Tensor::matmul invalid right tensor storage");

  Tensor result({a_rows, b_cols});

  // HYBRYDOWY SILNIK:
  // Jeśli macierz jest duża (np. faza przetwarzania całego promptu), używamy
  // GPU. Dla małych macierzy (np. generowanie pojedynczego tokena) używamy CPU
  // OpenMP, ponieważ kopiowanie danych z RAM do VRAM dla małych ilości danych
  // zajęłoby więcej czasu niż samo liczenie na CPU.
  bool tryGpu = a_rows >= 16;
#ifdef NPPAI_TESTING
  if (g_gpuOverride == 0)
    tryGpu = false;
  else if (g_gpuOverride == 1)
    tryGpu = true;
#endif
  if (tryGpu) {
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
#ifdef NPPAI_X86_SIMD
        if (useAVX2FMA()) {
          int k = 0;
          __m256 sum0 = _mm256_setzero_ps();
          __m256 sum1 = _mm256_setzero_ps();
          __m256 sum2 = _mm256_setzero_ps();
          __m256 sum3 = _mm256_setzero_ps();
          for (; k <= a_cols - 32; k += 32) {
            const __m256 va0 = _mm256_loadu_ps(&a.data[i * a.shape[1] + k]);
            const __m256 va1 = _mm256_loadu_ps(&a.data[i * a.shape[1] + k + 8]);
            const __m256 va2 = _mm256_loadu_ps(&a.data[i * a.shape[1] + k + 16]);
            const __m256 va3 = _mm256_loadu_ps(&a.data[i * a.shape[1] + k + 24]);
            const __m128i vbLo =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    &b.data_q8[j * b.shape[1] + k]));
            const __m128i vbHi =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    &b.data_q8[j * b.shape[1] + k + 16]));
            const __m256i vb0 = _mm256_cvtepi8_epi32(vbLo);
            const __m256i vb1 =
                _mm256_cvtepi8_epi32(_mm_srli_si128(vbLo, 8));
            const __m256i vb2 = _mm256_cvtepi8_epi32(vbHi);
            const __m256i vb3 =
                _mm256_cvtepi8_epi32(_mm_srli_si128(vbHi, 8));
            sum0 = _mm256_fmadd_ps(va0, _mm256_cvtepi32_ps(vb0), sum0);
            sum1 = _mm256_fmadd_ps(va1, _mm256_cvtepi32_ps(vb1), sum1);
            sum2 = _mm256_fmadd_ps(va2, _mm256_cvtepi32_ps(vb2), sum2);
            sum3 = _mm256_fmadd_ps(va3, _mm256_cvtepi32_ps(vb3), sum3);
          }
          for (; k <= a_cols - 16; k += 16) {
            const __m256 va0 = _mm256_loadu_ps(&a.data[i * a.shape[1] + k]);
            const __m256 va1 = _mm256_loadu_ps(&a.data[i * a.shape[1] + k + 8]);
            const __m128i vb16 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    &b.data_q8[j * b.shape[1] + k]));
            const __m256i vb0 = _mm256_cvtepi8_epi32(vb16);
            const __m256i vb1 =
                _mm256_cvtepi8_epi32(_mm_srli_si128(vb16, 8));
            sum0 = _mm256_fmadd_ps(va0, _mm256_cvtepi32_ps(vb0), sum0);
            sum1 = _mm256_fmadd_ps(va1, _mm256_cvtepi32_ps(vb1), sum1);
          }
          for (; k <= a_cols - 8; k += 8) {
            const __m256 va = _mm256_loadu_ps(&a.data[i * a.shape[1] + k]);
            const __m128i vb8 =
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                    &b.data_q8[j * b.shape[1] + k]));
            const __m256i vb = _mm256_cvtepi8_epi32(vb8);
            sum0 = _mm256_fmadd_ps(va, _mm256_cvtepi32_ps(vb), sum0);
          }
          const __m256 sum01 = _mm256_add_ps(sum0, sum1);
          const __m256 sum23 = _mm256_add_ps(sum2, sum3);
          const __m256 sum_vec = _mm256_add_ps(sum01, sum23);
          const __m128 low = _mm256_castps256_ps128(sum_vec);
          const __m128 high = _mm256_extractf128_ps(sum_vec, 1);
          __m128 reduced = _mm_add_ps(low, high);
          reduced = _mm_hadd_ps(reduced, reduced);
          reduced = _mm_hadd_ps(reduced, reduced);
          float unscaled_sum = _mm_cvtss_f32(reduced);
          for (; k < a_cols; k++) {
            unscaled_sum +=
                a.data[i * a.shape[1] + k] *
                static_cast<float>(b.data_q8[j * b.shape[1] + k]);
          }
          sum = unscaled_sum * b.scale_q8;
        } else {
          for (int k = 0; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] *
                   (b.data_q8[j * b.shape[1] + k] * b.scale_q8);
          }
        }
#else
          for (int k = 0; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] * (b.data_q8[j * b.shape[1] + k] * b.scale_q8);
          }
#endif
        } else {
#ifdef NPPAI_X86_SIMD
        if (useAVX2FMA()) {
          int k = 0;
          __m256 sum_vec = _mm256_setzero_ps();
          for (; k <= a_cols - 8; k += 8) {
            __m256 va = _mm256_loadu_ps(&a.data[i * a.shape[1] + k]);
            __m256 vb = _mm256_loadu_ps(&b.data[j * b.shape[1] + k]);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
          }
          float tmp[8];
          _mm256_storeu_ps(tmp, sum_vec);
          for (int m = 0; m < 8; m++) sum += tmp[m];
          for (; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] * b.data[j * b.shape[1] + k];
          }
        } else {
          for (int k = 0; k < a_cols; k++) {
            sum += a.data[i * a.shape[1] + k] * b.data[j * b.shape[1] + k];
          }
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
            sum += a.data[i * a.shape[1] + k] * (b.data_q8[k * b.shape[1] + j] * b.scale_q8);
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

#ifdef NPPAI_X86_SIMD
    if (useAVX2FMA()) {
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
    }
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
#ifdef NPPAI_X86_SIMD
    if (useAVX2FMA()) {
    // Faza 2: Normalizacja z AVX2
    __m256 ss_vec = _mm256_set1_ps(ss);
    for (; c <= cols - 8; c += 8) {
      __m256 val = _mm256_loadu_ps(&data[r * cols + c]);
      __m256 w = _mm256_loadu_ps(&weight.data[c]);
      __m256 res = _mm256_mul_ps(_mm256_mul_ps(val, ss_vec), w);
      _mm256_storeu_ps(&data[r * cols + c], res);
    }
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

bool Tensor::readFromFile(std::ifstream &file, bool quantize) {
  if (!file.is_open() || !file.good()) return false;
  const std::streamsize bytes = static_cast<std::streamsize>(data.size() * sizeof(float));
  std::vector<float> loaded(data.size());
  file.read(reinterpret_cast<char *>(loaded.data()), bytes);
  if (file.gcount() != bytes || !file) return false;
  for (float value : loaded) if (!std::isfinite(value)) return false;
  if (!quantize) { data = std::move(loaded); data_q8.clear(); scale_q8 = 0.0f; return true; }
  float max_abs = 0.0f;
  for (float value : loaded) max_abs = (std::max)(max_abs, std::abs(value));
  const float newScale = max_abs == 0.0f ? 1e-9f : max_abs / 127.0f;
  std::vector<int8_t> quantized(loaded.size());
  for (size_t i = 0; i < loaded.size(); ++i) {
    const float scaled = std::round(loaded[i] / newScale);
    quantized[i] = static_cast<int8_t>((std::max)(-127.0f, (std::min)(127.0f, scaled)));
  }
  data_q8 = std::move(quantized); scale_q8 = newScale;
  data.clear(); data.shrink_to_fit(); return true;
}

// --- MOE ROUTING ---
std::vector<MoERoute> MoERouter::topK(const std::vector<float>& logits, int k) {
  if (k <= 0 || logits.empty() || static_cast<size_t>(k) > logits.size())
    throw std::invalid_argument("MoERouter::topK invalid expert count");

  std::vector<int> indices;
  indices.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i]))
      throw std::invalid_argument("MoERouter::topK requires finite logits");
    indices.push_back(static_cast<int>(i));
  }

  std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
    if (logits[a] != logits[b]) return logits[a] > logits[b];
    return a < b;
  });
  indices.resize(static_cast<size_t>(k));

  float maxLogit = logits[indices[0]];
  double denominator = 0.0;
  for (int index : indices)
    denominator += std::exp(static_cast<double>(logits[index] - maxLogit));

  std::vector<MoERoute> routes;
  routes.reserve(indices.size());
  for (int index : indices) {
    const double numerator = std::exp(static_cast<double>(logits[index] - maxLogit));
    routes.push_back({index, static_cast<float>(numerator / denominator), false});
  }
  return routes;
}

std::vector<MoERoute> MoERouter::topKWithCapacity(
    const std::vector<float>& logits, int k, std::vector<int>& expertLoads,
    int capacityPerExpert) {
  if (capacityPerExpert <= 0 || expertLoads.size() != logits.size())
    throw std::invalid_argument("MoERouter::topKWithCapacity invalid capacity state");
  for (int load : expertLoads)
    if (load < 0 || load > capacityPerExpert)
      throw std::invalid_argument("MoERouter::topKWithCapacity invalid expert load");

  // Request a full ranking so fallback selection obeys exactly the same
  // deterministic ordering as topK.
  const auto ranked = topK(logits, static_cast<int>(logits.size()));
  std::vector<MoERoute> selected;
  selected.reserve(static_cast<size_t>(k));
  for (const auto& candidate : ranked) {
    if (expertLoads[candidate.expertIndex] < capacityPerExpert) {
      const bool fallback = selected.size() >= static_cast<size_t>(k) ? true : false;
      selected.push_back({candidate.expertIndex, candidate.weight, fallback});
      if (selected.size() == static_cast<size_t>(k)) break;
    }
  }
  if (selected.size() != static_cast<size_t>(k))
    throw std::runtime_error("MoERouter::topKWithCapacity insufficient expert capacity");

  // A route is a fallback when a higher-ranked expert was skipped because it
  // was full. Re-normalize only the experts that will actually execute.
  bool skippedFull = false;
  size_t selectedPos = 0;
  for (const auto& candidate : ranked) {
    if (selectedPos == selected.size()) break;
    if (candidate.expertIndex == selected[selectedPos].expertIndex) {
      selected[selectedPos].usedFallback = skippedFull;
      ++selectedPos;
    } else if (expertLoads[candidate.expertIndex] >= capacityPerExpert) {
      skippedFull = true;
    }
  }

  float maxLogit = logits[selected[0].expertIndex];
  double denominator = 0.0;
  for (const auto& route : selected)
    denominator += std::exp(static_cast<double>(logits[route.expertIndex] - maxLogit));
  for (auto& route : selected) {
    route.weight = static_cast<float>(
        std::exp(static_cast<double>(logits[route.expertIndex] - maxLogit)) / denominator);
    ++expertLoads[route.expertIndex];
  }
  return selected;
}

// --- ENGINE IMPLEMENTATION ---
NppAIEngine::NppAIEngine() {}

NppAIEngine::~NppAIEngine() {}

bool NppAIEngine::loadModel(const std::string &modelPath) {
  std::lock_guard<std::mutex> lock(engineMutex);
  std::ifstream file(modelPath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Nie udalo sie otworzyc pliku modelu: " << modelPath << std::endl;
    return false;
  }

  constexpr char kMagic[8] = {'N','P','P','A','I','\0','\0','\0'};
  constexpr uint32_t kFormatVersion = 2;
  constexpr std::streamoff kV2HeaderBytes = 8 + 4 + 5 * 4 + 8 + 32;

  char magic[8] = {};
  file.read(magic, sizeof(magic));
  const bool isV2 = file.gcount() == static_cast<std::streamsize>(sizeof(magic)) &&
                    std::memcmp(magic, kMagic, sizeof(magic)) == 0;
  file.clear();
  file.seekg(0, std::ios::beg);

  int32_t header[5] = {};
  uint64_t declaredPayloadBytes = 0;
  std::array<unsigned char, 32> declaredSha256{};
  std::streamoff payloadOffset = 0;

  if (isV2) {
    file.read(magic, sizeof(magic));
    uint32_t version = 0;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(header), sizeof(header));
    file.read(reinterpret_cast<char *>(&declaredPayloadBytes), sizeof(declaredPayloadBytes));
    file.read(reinterpret_cast<char *>(declaredSha256.data()), declaredSha256.size());
    if (!file || version != kFormatVersion) {
      std::cerr << "Nieobslugiwana wersja lub niepelny naglowek modelu NppAI.\n";
      return false;
    }
    payloadOffset = kV2HeaderBytes;
  } else {
    // Legacy v1: five int32 dimensions followed directly by FP32 tensors.
    file.read(reinterpret_cast<char *>(header), sizeof(header));
    if (file.gcount() != static_cast<std::streamsize>(sizeof(header))) {
      std::cerr << "Nieprawidlowy lub niepelny naglowek modelu.\n";
      return false;
    }
    payloadOffset = sizeof(header);
  }

  const int modelDim = header[0];
  const int modelHiddenDim = header[1];
  const int modelLayers = header[2];
  const int modelMaxSeqLen = header[3];
  const int modelVocabSize = header[4];

  constexpr int kMaxDimension = 1 << 15;
  constexpr int kMaxLayers = 256;
  if (modelDim <= 0 || modelHiddenDim <= 0 || modelVocabSize <= 0 ||
      modelMaxSeqLen <= 0 || modelLayers <= 0 ||
      modelDim > kMaxDimension || modelHiddenDim > kMaxDimension ||
      modelVocabSize > kMaxDimension * 16 ||
      modelMaxSeqLen > kMaxDimension || modelLayers > kMaxLayers) {
    std::cerr << "Nieprawidlowe wymiary modelu.\n";
    return false;
  }

  file.seekg(0, std::ios::end);
  const std::streamoff fileSize = file.tellg();
  if (fileSize < payloadOffset) {
    std::cerr << "Nieprawidlowy rozmiar pliku modelu.\n";
    return false;
  }

  const uint64_t d = static_cast<uint64_t>(modelDim);
  const uint64_t h = static_cast<uint64_t>(modelHiddenDim);
  const uint64_t v = static_cast<uint64_t>(modelVocabSize);
  const uint64_t t = static_cast<uint64_t>(modelMaxSeqLen);
  const uint64_t l = static_cast<uint64_t>(modelLayers);
  const uint64_t floatCount =
      v * d + t * d +
      l * (2ULL * d + 4ULL * d * d + 3ULL * d * h) +
      d + d * v;
  constexpr uint64_t kMaxModelBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  const uint64_t expectedPayloadBytes = floatCount * sizeof(float);
  if (floatCount > (UINT64_MAX / sizeof(float)) ||
      expectedPayloadBytes > kMaxModelBytes ||
      static_cast<uint64_t>(fileSize - payloadOffset) != expectedPayloadBytes ||
      (isV2 && declaredPayloadBytes != expectedPayloadBytes)) {
    std::cerr << "Model przekracza limit rozmiaru lub ma nieprawidlowy payload.\n";
    return false;
  }

  if (isV2) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD objectBytes = 0, resultBytes = 0;
    std::vector<unsigned char> hashObject;
    std::array<unsigned char, 32> actualSha256{};
    bool hashOk = false;

    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) == 0 &&
        BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                          reinterpret_cast<PUCHAR>(&objectBytes), sizeof(objectBytes),
                          &resultBytes, 0) == 0) {
      hashObject.resize(objectBytes);
      if (BCryptCreateHash(algorithm, &hash, hashObject.data(), objectBytes,
                           nullptr, 0, 0) == 0) {
        file.clear();
        file.seekg(payloadOffset, std::ios::beg);
        std::array<unsigned char, 64 * 1024> buffer{};
        uint64_t remaining = expectedPayloadBytes;
        hashOk = true;
        while (remaining > 0) {
          const std::streamsize chunk = static_cast<std::streamsize>(
              (std::min)(remaining, static_cast<uint64_t>(buffer.size())));
          file.read(reinterpret_cast<char *>(buffer.data()), chunk);
          if (file.gcount() != chunk ||
              BCryptHashData(hash, buffer.data(), static_cast<ULONG>(chunk), 0) != 0) {
            hashOk = false;
            break;
          }
          remaining -= static_cast<uint64_t>(chunk);
        }
        if (hashOk && BCryptFinishHash(hash, actualSha256.data(),
                                       static_cast<ULONG>(actualSha256.size()), 0) != 0)
          hashOk = false;
      }
    }
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);

    if (!hashOk || actualSha256 != declaredSha256) {
      std::cerr << "SHA-256 payloadu modelu nie zgadza sie z naglowkiem.\n";
      return false;
    }
  }

  file.clear();
  file.seekg(payloadOffset, std::ios::beg);

  dim = modelDim;
  hidden_dim = modelHiddenDim;
  n_layers = modelLayers;
  max_seq_len = modelMaxSeqLen;
  vocab_size = modelVocabSize;

  tokenEmbeddingTable = Tensor({vocab_size, dim});
  if (!tokenEmbeddingTable.readFromFile(file, false)) return false;
  posEmbeddingTable = Tensor({max_seq_len, dim});
  if (!posEmbeddingTable.readFromFile(file, false)) return false;

  layers.clear();
  for (int i = 0; i < n_layers; i++) {
    TransformerLayer layer;
    layer.rmsAttn = Tensor({dim});
    if (!layer.rmsAttn.readFromFile(file, false)) return false;
    layer.wQ = Tensor({dim, dim});
    if (!layer.wQ.readFromFile(file, true)) return false;
    layer.wK = Tensor({dim, dim});
    if (!layer.wK.readFromFile(file, true)) return false;
    layer.wV = Tensor({dim, dim});
    if (!layer.wV.readFromFile(file, true)) return false;
    layer.wO = Tensor({dim, dim});
    if (!layer.wO.readFromFile(file, true)) return false;
    layer.rmsFFN = Tensor({dim});
    if (!layer.rmsFFN.readFromFile(file, false)) return false;
    layer.wGate = Tensor({dim, hidden_dim});
    if (!layer.wGate.readFromFile(file, true)) return false;
    layer.wUp = Tensor({dim, hidden_dim});
    if (!layer.wUp.readFromFile(file, true)) return false;
    layer.wDown = Tensor({hidden_dim, dim});
    if (!layer.wDown.readFromFile(file, true)) return false;
    layers.push_back(layer);
  }

  outputRMSNorm = Tensor({dim});
  if (!outputRMSNorm.readFromFile(file, false)) return false;
  outputClassifier = Tensor({dim, vocab_size});
  if (!outputClassifier.readFromFile(file, true)) return false;

  file.close();
  std::string bpePath =
      modelPath.substr(0, modelPath.find_last_of("/\\")) + "\\bpe_merges.txt";
  loadBPETokenizer(bpePath);
  return true;
}

bool NppAIEngine::loadBPETokenizer(const std::string &path) {
  bpe_merges.clear();
  bpe_merge_ranks.clear();
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
  size_t mergeRank = 0;
  while (file >> p0 >> p1 >> idx) {
    // IDs below 256 are reserved for raw bytes. Every merge must reference
    // already-known tokens and create a new token ID.
    if (p0 < 0 || p1 < 0 || idx < 256 ||
        !bpe_vocab.count(p0) || !bpe_vocab.count(p1) ||
        bpe_vocab.count(idx) || bpe_merges.count({p0, p1})) {
      bpe_merges.clear();
      bpe_merge_ranks.clear();
      bpe_vocab.clear();
      for (int i = 0; i < 256; i++) {
        bpe_vocab[i] = std::string(1, (char)i);
      }
      std::cerr << "Nieprawidlowa definicja merge w BPE: " << path << "\n";
      return false;
    }

    bpe_merges[{p0, p1}] = idx;
    bpe_merge_ranks[{p0, p1}] = mergeRank++;
    bpe_vocab[idx] = bpe_vocab[p0] + bpe_vocab[p1];
  }

  if (!file.eof() && file.fail()) {
    bpe_merges.clear();
    bpe_merge_ranks.clear();
    bpe_vocab.clear();
    for (int i = 0; i < 256; i++) {
      bpe_vocab[i] = std::string(1, (char)i);
    }
    std::cerr << "Nieprawidlowy format pliku BPE: " << path << "\n";
    return false;
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
    std::pair<int, int> best_pair;
    size_t min_rank = (std::numeric_limits<size_t>::max)();

    for (size_t i = 0; i < ids.size() - 1; i++) {
      std::pair<int, int> pair = {ids[i], ids[i + 1]};
      auto rankIt = bpe_merge_ranks.find(pair);
      if (rankIt != bpe_merge_ranks.end() && rankIt->second < min_rank) {
        min_rank = rankIt->second;
        best_pair = pair;
      }
    }

    if (min_rank == (std::numeric_limits<size_t>::max)())
      break;

    std::vector<int> new_ids;
    for (size_t i = 0; i < ids.size(); i++) {
      if (i < ids.size() - 1 && ids[i] == best_pair.first &&
          ids[i + 1] == best_pair.second) {
        new_ids.push_back(bpe_merges.at(best_pair));
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