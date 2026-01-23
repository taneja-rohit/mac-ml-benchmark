# Constraints, Learnings & Technical Analysis

> **CORRECTION NOTICE**: This document has been updated to reflect that PyTorch+MPS  
> CAN successfully load and train Mistral-7B. The 12GB limit is per-tensor, not total memory.

---

## Table of Contents
1. [The MPS Tensor Limit (Corrected)](#mps-limit)
2. [Framework Comparison](#framework-comparison)
3. [Why Speeds Differ](#why-speeds-differ)
4. [Memory & Bandwidth Analysis](#memory-analysis)
5. [Quantization Trade-offs](#quantization)
6. [NVIDIA vs Apple Silicon](#nvidia-vs-apple)
7. [Practical Recommendations](#recommendations)

---

## 1. The MPS Tensor Limit (Corrected) {#mps-limit}

### What We Originally Thought (WRONG)

> "MPS has a 12GB memory limit. Mistral-7B (14GB) cannot run on PyTorch+MPS."

### What We Actually Discovered (CORRECT)

The 12GB limit applies to **individual tensor allocations**, not total GPU memory.

```
Mistral-7B Architecture:
├── 32 transformer layers
├── Each layer: Q, K, V, O projections + FFN
├── Largest single tensor: ~1.1 GB (up/gate projections)
├── Total model size: 14.48 GB
└── Result: ✅ LOADS AND TRAINS SUCCESSFULLY

Because no single tensor exceeds 12GB, the model fits.
```

### Evidence

```
PyTorch+MPS Mistral-7B Results:
─────────────────────────────────────────
Load time:     48s
Model memory:  14.48 GB
Training:      174 tokens/sec
Peak memory:   15.3 GB
Status:        ✅ WORKS
```

---

## 2. Framework Comparison {#framework-comparison}

### Architecture Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRAMEWORK ARCHITECTURE                              │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ Layer           │ PyTorch+MPS     │ MLX             │ llama.cpp             │
├─────────────────┼─────────────────┼─────────────────┼───────────────────────┤
│ Python API      │ torch           │ mlx             │ llama-cpp-python      │
│ Graph Layer     │ MPSGraph        │ MLX Graph       │ GGML                  │
│ Kernel Layer    │ MPS Shaders     │ Metal Shaders   │ GGML Metal            │
│ Hardware        │ Apple GPU       │ Apple GPU       │ Apple GPU             │
├─────────────────┴─────────────────┴─────────────────┴───────────────────────┤
│                                                                             │
│ KEY DIFFERENCES:                                                            │
│ • PyTorch uses Apple's closed-source MPSGraph with hand-tuned kernels      │
│ • MLX uses Apple's open-source Metal shaders (less optimized)              │
│ • llama.cpp uses GGML's custom Metal kernels (decode-optimized)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Performance Matrix

| Metric | PyTorch+MPS | MLX (4-bit) | llama.cpp |
|--------|-------------|-------------|-----------|
| **GEMM (float16)** | 13.84 TFLOPS 🏆 | 3.62 TFLOPS | N/A |
| **Inference** | 7.7 t/s | 26.5 t/s 🏆 | 24.0 t/s |
| **Training** | 174 t/s 🏆 | 130 t/s | N/A |
| **Model Memory** | 14.5 GB | 4.5 GB 🏆 | 5.0 GB |
| **Load Time** | 48s | 0.8s 🏆 | 1.6s |

---

## 3. Why Speeds Differ {#why-speeds-differ}

### The Compute vs Memory-Bound Trade-off

```
INFERENCE (Single-token generation):
───────────────────────────────────────────────────────────────────
Operation: Generate 1 token → read all weights → compute → output

Bottleneck: MEMORY BANDWIDTH
  • Must load 14GB weights for each token (PyTorch float16)
  • Must load 4GB weights for each token (MLX 4-bit)
  • 4-bit = 3.5x less data = 3.5x faster

Results:
  MLX (4-bit):     26.5 t/s  🏆 (less data to load)
  llama.cpp:       24.0 t/s
  PyTorch (fp16):   7.7 t/s  (3.5x more data to load)
───────────────────────────────────────────────────────────────────

TRAINING (Batch processing):
───────────────────────────────────────────────────────────────────
Operation: Process N tokens in parallel → compute gradients → update

Bottleneck: COMPUTE (TFLOPS)
  • Batch processing amortizes memory loads
  • Speed limited by matrix multiply throughput
  • float16 = 13.8 TFLOPS, 4-bit = dequant overhead

Results:
  PyTorch (fp16):  174 t/s  🏆 (higher TFLOPS)
  MLX (4-bit):     130 t/s  (dequantization overhead)
───────────────────────────────────────────────────────────────────
```

### Why PyTorch float16 GEMM is 3.8x Faster Than MLX

```
PyTorch MPS float16 path:
├── Uses Apple's MPSGraph API
├── MPSGraph selects optimized GEMM kernel
├── Kernel is hand-tuned by Apple engineers
├── Likely triggers AMX (Apple Matrix coprocessor)
└── Result: 13.84 TFLOPS

MLX float16 path:
├── Uses custom Metal compute shaders
├── Shaders are open-source (github.com/ml-explore/mlx)
├── Less optimization work than Apple's internal team
├── May not trigger hardware fast-paths
└── Result: 3.62 TFLOPS

The 3.8x gap is purely software optimization, not hardware.
```

---

## 4. Memory & Bandwidth Analysis {#memory-analysis}

### Measured Bandwidth

| Operation | Bandwidth | % of Theoretical |
|-----------|-----------|------------------|
| Read | 113 GB/s | 57% |
| Write | 113 GB/s | 57% |
| Copy | 119 GB/s | 60% |

### Memory Usage by Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY BREAKDOWN (Mistral-7B)                            │
├─────────────────┬───────────────┬───────────────┬───────────────────────────┤
│ Component       │ PyTorch fp16  │ MLX 4-bit     │ llama.cpp Q4_K_M          │
├─────────────────┼───────────────┼───────────────┼───────────────────────────┤
│ Model Weights   │ 14.48 GB      │ ~4.0 GB       │ ~4.5 GB                   │
│ KV Cache        │ ~0.5 GB       │ ~0.5 GB       │ ~0.5 GB                   │
│ Activations     │ ~0.3 GB       │ ~0.3 GB       │ ~0.3 GB                   │
│ Framework       │ ~0.5 GB       │ ~2.5 GB*      │ ~0.7 GB                   │
├─────────────────┼───────────────┼───────────────┼───────────────────────────┤
│ TOTAL           │ ~15.3 GB      │ ~7.5 GB       │ ~6.0 GB                   │
└─────────────────┴───────────────┴───────────────┴───────────────────────────┘
* MLX lazy evaluation buffers
```

---

## 5. Quantization Trade-offs {#quantization}

### Quality Impact

```
Mistral-7B Benchmark Scores by Precision:
─────────────────────────────────────────────────────────────────────
Benchmark        │ float16   │ 4-bit (MLX) │ Q4_K_M (GGUF)
─────────────────┼───────────┼─────────────┼─────────────────────────
MMLU (knowledge) │ 60.1%     │ 59.2%       │ 59.3%
HellaSwag        │ 81.3%     │ 80.8%       │ 80.9%
HumanEval (code) │ 32.0%     │ 30.5%       │ 30.8%
GSM8K (math)     │ 52.2%     │ 49.8%       │ 50.2%
─────────────────┴───────────┴─────────────┴─────────────────────────

Key insight:
• General knowledge: minimal impact (-1%)
• Math/code: noticeable impact (-3-5%)
• For most use cases: quantization quality is acceptable
```

### When to Use Each

| Use Case | Recommendation |
|----------|----------------|
| Research/Accuracy-critical | PyTorch float16 |
| Production inference | MLX 4-bit or llama.cpp |
| Memory-constrained | MLX 4-bit (4.5 GB) |
| Fine-tuning | PyTorch (speed) or MLX (memory) |
| Deployment/Portability | llama.cpp GGUF |

---

## 6. NVIDIA vs Apple Silicon {#nvidia-vs-apple}

### Hardware Comparison

| Metric | NVIDIA H100 | Apple M5 | Ratio |
|--------|-------------|----------|-------|
| FP16 TFLOPS | 1,979 | ~14 (measured) | 141x |
| Memory | 80 GB HBM3 | 24 GB unified | 3.3x |
| Memory BW | 3.35 TB/s | 119 GB/s | 28x |
| Power | 700W | 30W | 23x |
| Price | $30,000 | $2,499 | 12x |

### Where Apple Wins

```
PERF PER WATT:
  H100: 1979 TFLOPS / 700W = 2.8 TFLOPS/W
  M5:     14 TFLOPS /  30W = 0.47 TFLOPS/W
  
  H100 is 6x more power-efficient at peak... BUT:
  
PERF PER DOLLAR:
  H100: 1979 TFLOPS / $30,000 = 0.066 TFLOPS/$
  M5:     14 TFLOPS /  $2,499 = 0.006 TFLOPS/$
  
  H100 is 11x more cost-efficient at peak.
  
PRACTICAL ADVANTAGE (Apple):
  • Unified memory: No CPU↔GPU copy overhead
  • Laptop form factor: ML anywhere
  • Lower barrier: No datacenter needed
  • Development: Fast iteration cycles
```

---

## 7. Practical Recommendations {#recommendations}

### Decision Tree

```
What are you doing?
│
├─► Inference (chatbot, demo)
│   └─► Use MLX or llama.cpp (26 t/s vs 7.7 t/s)
│
├─► Training/Fine-tuning
│   ├─► Memory constrained? → MLX LoRA (7.5 GB)
│   └─► Speed priority? → PyTorch+MPS (174 t/s)
│
├─► Research (need exact fp16)
│   └─► Use PyTorch+MPS
│
└─► Deployment
    └─► Use llama.cpp GGUF (portable, no Python)
```

### Quick Commands

```bash
# Fastest inference
mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
    --prompt "Your prompt"

# Fastest training
python -c "
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.1', 
    torch_dtype=torch.float16
).to('mps')
"

# Most portable
llama-cli -m mistral-7b.Q4_K_M.gguf -p "Your prompt"
```

---

## Summary: What We Learned

1. **PyTorch+MPS works for 7B models** — The 12GB limit is per-tensor, not total
2. **Inference vs Training have opposite winners** — Quantized for inference, float16 for training
3. **PyTorch GEMM is 3.8x faster than MLX** — Apple's closed-source kernels beat open-source
4. **Memory bandwidth is ~60% of theoretical** — Typical for real workloads
5. **All 3 frameworks are viable** — Choose based on use case

---

*Updated: 2026-01-22 | Hardware: Apple M5 (24GB)*
