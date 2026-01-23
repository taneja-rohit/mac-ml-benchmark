# Mac ML Benchmark Results — Apple M5 (24GB)

> **CORRECTED**: Earlier documentation incorrectly stated PyTorch+MPS cannot run Mistral-7B.  
> This was wrong. PyTorch+MPS **CAN** load and train Mistral-7B in float16.

---

## Executive Summary

| Framework | Model Memory | Inference Speed | Training Speed | Status |
|-----------|--------------|-----------------|----------------|--------|
| **PyTorch+MPS** | 14.5 GB | 7.7 t/s | **174 t/s** 🏆 | ✅ WORKS |
| **MLX** (4-bit) | 4.5 GB | **26.5 t/s** 🏆 | 130 t/s | ✅ WORKS |
| **llama.cpp** (GGUF) | 5.0 GB | 24.0 t/s | N/A | ✅ WORKS |

**Key Insight**: PyTorch+MPS is fastest for training (batch processing), while quantized frameworks (MLX, llama.cpp) are faster for inference (single-token generation).

---

## Hardware Specifications

```
Apple M5 (MacBook Pro)
├── CPU Cores: 10 (4 Performance + 6 Efficiency)
├── GPU Cores: 12
├── Unified Memory: 24 GB
├── Memory Bandwidth: 119 GB/s (measured)
├── Theoretical Peak: ~200 GB/s
└── MPS Available: ✅
```

---

## 1. Compute Benchmarks (GEMM)

Matrix multiplication performance at different sizes and precisions:

### PyTorch + MPS

| Size | float32 | float16 |
|------|---------|---------|
| 1024×1024 | 2.59 TFLOPS | 6.32 TFLOPS |
| 2048×2048 | 3.43 TFLOPS | 12.62 TFLOPS |
| 4096×4096 | 3.16 TFLOPS | **13.84 TFLOPS** 🏆 |

### MLX

| Size | float32 | float16 |
|------|---------|---------|
| 1024×1024 | 2.58 TFLOPS | 2.67 TFLOPS |
| 2048×2048 | 3.32 TFLOPS | 3.59 TFLOPS |
| 4096×4096 | 3.25 TFLOPS | 3.62 TFLOPS |

### Analysis

```
GEMM SPEEDUP (PyTorch vs MLX):
─────────────────────────────────────────
Precision │ PyTorch │ MLX    │ Ratio
─────────────────────────────────────────
float32   │ 3.16 TF │ 3.25 TF│ 0.97x (same)
float16   │ 13.84 TF│ 3.62 TF│ 3.82x 🏆
─────────────────────────────────────────

WHY? PyTorch MPS uses Apple's MPSGraph with hand-tuned float16
GEMM kernels. MLX uses open-source Metal compute shaders that
haven't been optimized to the same degree.
```

---

## 2. Memory Bandwidth

| Operation | Bandwidth | Notes |
|-----------|-----------|-------|
| Sequential Read | 113 GB/s | Sum reduction |
| Sequential Write | 113 GB/s | Fill operation |
| Copy (R+W) | **119 GB/s** | Best achieved |
| Theoretical | ~200 GB/s | Apple spec |
| **Utilization** | **~60%** | Typical for real workloads |

---

## 3. Model Benchmarks (Mistral-7B)

### PyTorch + MPS (float16)

```
Model: mistralai/Mistral-7B-v0.1
Precision: float16
Quantization: None

LOAD:
  Time: 48.0s (includes download caching)
  Model Memory: 14.48 GB

INFERENCE:
  Tokens/sec: 7.7 t/s
  Peak Memory: 14.48 GB
  
TRAINING (LoRA-style):
  Tokens/sec: 174 t/s 🏆
  Peak Memory: 15.3 GB
  Loss convergence: ✅ (requires gradient clipping for stability)
```

### MLX (4-bit)

```
Model: mlx-community/Mistral-7B-Instruct-v0.2-4bit
Precision: 4-bit quantized
Quantization: Native MLX

LOAD:
  Time: 0.8s (cached)
  Model Memory: ~4.5 GB

INFERENCE:
  Tokens/sec: 26.5 t/s 🏆
  Peak Memory: ~7.5 GB
  
TRAINING (LoRA):
  Tokens/sec: 122-133 t/s
  Peak Memory: 7.5 GB
  Loss convergence: ✅
```

### llama.cpp (GGUF Q4_K_M)

```
Model: Mistral-7B-Instruct-v0.3-GGUF
Precision: Q4_K_M (mixed 4/6-bit)
Quantization: GGUF k-quant

LOAD:
  Time: 1.6s (cached)
  Model Memory: ~5.0 GB

INFERENCE:
  Tokens/sec: 24.0 t/s
  Peak Memory: ~6.0 GB
  
TRAINING:
  Not supported
```

---

## 4. Why Are Speeds Different?

### The Compute vs Memory-Bound Trade-off

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE vs TRAINING                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  INFERENCE (Single-token generation):                                      │
│  ─────────────────────────────────────                                     │
│  • Generate 1 token at a time                                              │
│  • Small batch size (typically 1)                                          │
│  • MEMORY-BOUND: Speed limited by weight loading, not compute              │
│  • 4-bit quantization wins: 3x less data to load per token                 │
│                                                                            │
│  Winner: MLX (26.5 t/s) > llama.cpp (24 t/s) > PyTorch (7.7 t/s)          │
│                                                                            │
│  TRAINING (Batch processing):                                              │
│  ─────────────────────────────────────                                     │
│  • Process many tokens in parallel                                         │
│  • Larger batch sizes possible                                             │
│  • COMPUTE-BOUND: Speed limited by TFLOPS, not memory                      │
│  • float16 wins: Higher arithmetic throughput (13.8 vs 3.6 TFLOPS)         │
│                                                                            │
│  Winner: PyTorch (174 t/s) > MLX (130 t/s) > llama.cpp (N/A)              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Framework Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FRAMEWORK DEEP DIVE                                      │
├─────────────────┬───────────────┬───────────────┬───────────────────────────┤
│                 │ PyTorch+MPS   │ MLX           │ llama.cpp                 │
├─────────────────┼───────────────┼───────────────┼───────────────────────────┤
│ Metal Backend   │ MPSGraph      │ Custom Shaders│ GGML Metal                │
│ Execution       │ Eager         │ Lazy          │ Eager+Batched             │
│ Memory Model    │ CUDA-style    │ Unified-aware │ Manual management         │
│ GEMM Kernels    │ Apple-tuned   │ Open-source   │ GGML hand-written         │
│ Quantization    │ Limited       │ Native 4-bit  │ GGUF (sophisticated)      │
│ Training        │ ✅ Full       │ ✅ LoRA       │ ❌ No                      │
│ Ecosystem       │ HuggingFace   │ mlx-community │ Standalone                │
├─────────────────┴───────────────┴───────────────┴───────────────────────────┤
│                                                                             │
│ WHY PYTORCH INFERENCE IS SLOW:                                              │
│ • HuggingFace generate() is not optimized for single-token decode          │
│ • Each token requires a full forward pass with framework overhead          │
│ • No KV-cache optimization in standard transformers                        │
│ • float16 = 2x memory bandwidth vs 4-bit = slower token loading            │
│                                                                             │
│ WHY MLX INFERENCE IS FAST:                                                  │
│ • Lazy evaluation batches operations across tokens                         │
│ • 4-bit = 4x less memory bandwidth per token                               │
│ • Optimized for Apple's unified memory architecture                        │
│ • KV-cache built into mlx-lm                                               │
│                                                                             │
│ WHY LLAMA.CPP IS CONSISTENT:                                                │
│ • Written in C with manual memory management                               │
│ • GGML: extremely optimized for decode-heavy workloads                     │
│ • Q4_K_M: uses 6-bit for critical weights, 4-bit for others               │
│ • Metal backend: direct GPU access without Python overhead                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. MPS Tensor Limit (Corrected Understanding)

### What We Originally Thought (WRONG)

> "MPS has a 12GB total memory limit. Mistral-7B (14GB) won't fit."

### What We Actually Found (CORRECT)

```
The 12GB limit is PER-TENSOR, not total GPU memory.

Mistral-7B has 32 layers × multiple tensors per layer.
Largest single tensor: ~1.1 GB (projection matrices)
Total model size: 14.48 GB

Since no single tensor exceeds 12GB, the model LOADS SUCCESSFULLY.

╔═══════════════════════════════════════════════════════════════════╗
║  CORRECTED UNDERSTANDING                                          ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   Physical RAM:         24 GB                                     ║
║   MPS per-tensor limit: 12 GB                                     ║
║   Mistral-7B total:     14.48 GB  ✅ FITS (distributed tensors)   ║
║   Largest tensor:       ~1.1 GB   ✅ Under limit                   ║
║                                                                   ║
║   Status: IT WORKS                                                ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 6. Recommendations

### For Inference (Chatbots, Demos)

**Use MLX or llama.cpp** — 3-4x faster than PyTorch for single-token generation.

```bash
# MLX (easiest)
mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.2-4bit --prompt "Hello"

# llama.cpp (most portable)
llama-cli -m mistral-7b.Q4_K_M.gguf -p "Hello"
```

### For Training/Fine-tuning

**Use PyTorch+MPS** — 30% faster than MLX for batch processing.

```python
# PyTorch (fastest training)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16).to("mps")
```

### For Memory-Constrained Scenarios

**Use MLX 4-bit** — Same quality, 3x less memory.

```bash
# 4.5 GB vs 14.5 GB
mlx_lm.lora --model mlx-community/Mistral-7B-Instruct-v0.2-4bit --train --data ./data
```

---

## 7. Raw Data Files

All benchmark data is stored in `results/raw/M5/`:

| File | Contents |
|------|----------|
| `system_info.json` | Hardware specifications |
| `pytorch_mps_benchmarks.json` | GEMM/Attention benchmarks |
| `pytorch_vs_mlx_detailed.json` | Side-by-side comparison |
| `memory_benchmarks.json` | Bandwidth measurements |
| `framework_comparison.json` | Full framework comparison |
| `pytorch_mps_mistral_finetune.json` | Training benchmark |

---

*Generated: 2026-01-22 on Apple M5 (24GB)*
