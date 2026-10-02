# NppAI Architecture

## Components

### NppAI plugin

The native plugin integrates with Notepad++ and exposes the AI functionality through the editor UI.

### NppAIEngine

The inference engine contains:

- tensor storage;
- Transformer layers;
- RMSNorm;
- attention;
- SwiGLU-style feed-forward blocks;
- tokenization;
- sampling;
- FP32 and selected INT8 weight handling;
- CPU and DirectX 11 matrix multiplication paths.

### RAGManager

Provides local project-context functionality. Its job is to collect/index relevant editor content and make retrieved context available to inference.

### TelemetryManager

Tracks correction information intended for the optional training pipeline.

### Python pipeline

Python/PyTorch is used for model training, model validation and export to the custom `.nppai` representation.

## Data flow

```text
Notepad++ editor
      │
      ▼
   AIManager
      │
      ├──────────────► RAGManager ─────► local context
      │
      ▼
 NppAIEngine
      │
      ├── CPU / OpenMP / AVX2
      │
      └── DirectX 11 compute
      │
      ▼
 generated tokens
      │
      ▼
 Notepad++ UI

Optional:

correction data
      │
      ▼
FastAPI backend
      │
      ▼
training dataset
      │
      ▼
PyTorch training
      │
      ▼
.nppai model
