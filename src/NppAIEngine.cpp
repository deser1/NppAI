#include "NppAIEngine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#include <intrin.h>

static bool cpuSupportsAVX2() {
  int cpuInfo[4] = {};
  __cpuid(cpuInfo, 0);
  if (cpuInfo[0] < 7)
    return false;
  __cpuidex(cpuInfo, 7, 0);
  return (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
}

#define USE_AVX2
#else
static bool cpuSupportsAVX2() { return false; }
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
#if defined(_M_X64) || defined(__x86_64__)
          if (cpuSupportsAVX2()) {
            int k = 0;
            __m256 sum_vec = _mm256_setzero_ps();
            __m256 scale_vec = _mm256_set1_ps(b.scale_q8);
            for (; k <= a_cols - 8; k += 8) {
              __m256 va = _mm256_loadu_ps(&a.data[i * a.shape[1] + k]);
              __m128i vb_int8 = _mm_loadl_epi64((__m128i*)&b.data_q8[j * b.shape[1] + k]);
              __m256i vb_int32 = _mm256_cvtepi8_epi32(vb_int8);
              __m256 vb_float = _mm256_cvtepi32_ps(vb_int32);
              vb_float = _mm256_mul_ps(vb_float, scale_vec);
              sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb_float));
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, sum_vec);
            for (int m = 0; m < 8; ++m) sum += tmp[m];
            for (; k < a_cols; ++k)
              sum += a.data[i * a.shape[1] + k] * (b.data_q8[j * b.shape[1] + k] * b.scale_q8);
          } else
#endif
          {
            for (int k = 0; k < a_cols; ++k)
              sum += a.data[i * a.shape[1] + k] * (b.data_q8[j * b.shape[1] + k] * b.scale_q8);
          }
        } else {
#if defined(_M_X64) || defined(__x86_64__)
          if (cpuSupportsAVX2()) {
            int k = 0;
            __m256 sum_vec = _mm256_setzero_ps();
            for (; k <= a_cols - 8; k += 8) {
              __m256 va = _mm256_loadu_ps(&a.data[i * a.shape[1] + k]);
              __m256 vb = _mm256_loadu_ps(&b.data[j * b.shape[1] + k]);
              sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, sum_vec);
            for (int m = 0; m < 8; ++m) sum += tmp[m];
            for (; k < a_cols; ++k)
              sum += a.data[i * a.shape[1] + k] * b.data[j * b.shape[1] + k];
          } else
#endif
          {
            for (int k = 0; k < a_cols; ++k)
              sum += a.data[i * a.shape[1] + k] * b.data[j * b.shape[1] + k];
          }
        }
      } else {
        if (!b.data_q8.empty()) {
          for (int k = 0; k < a_cols; ++k)
            sum += a.data[i * a.shape[1] + k] * (b.data_q8[k * b.shape[1] + j] * b.scale_q8);
        } else {
          for (int k = 0; k < a_cols; ++k)
            sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
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

#if defined(_M_X64) || defined(__x86_64__)
    if (cpuSupportsAVX2()) {
      __m256 sum_vec = _mm256_setzero_ps();
      for (; c <= cols - 8; c += 8) {
        __m256 val = _mm256_loadu_ps(&data[r * cols + c]);
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(val, val));
      }
      float tmp[8];
      _mm256_storeu_ps(tmp, sum_vec);
      for (int i = 0; i < 8; i++) ss += tmp[i];
    }
#endif

    for (; c < cols; c++) {
      float val = data[r * cols + c];
      ss += val * val;
    }

    ss /= cols;
    ss += 1e-5f;
    ss = 1.0f / std::sqrt(ss);

    c = 0;
#if defined(_M_X64) || defined(__x86_64__)
    if (cpuSupportsAVX2()) {
      __m256 ss_vec = _mm256_set1_ps(ss);
      for (; c <= cols - 8; c += 8) {
        __m256 val = _mm256_loadu_ps(&data[r * cols + c]);
        __m256 w = _mm256_loadu_ps(&weight.data[c]);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(val, ss_vec), w);
        _mm256_storeu_ps(&data[r * cols + c], res);
      }
    }
#endif

    for (; c < cols; c++)
      data[r * cols + c] = (data[r * cols + c] * ss) * weight.data[c];
  }
}
}