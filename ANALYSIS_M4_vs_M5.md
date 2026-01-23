# M4 Pro vs M5 Benchmark Analysis

**Date:** January 22, 2026  
**Precision:** Float16 (all comparisons)

---

## Hardware Specs

| Spec | M5 | M4 Pro |
|------|-----|--------|
| CPU Cores | 10 (4P + 6E) | 12 (8P + 4E) |
| GPU Cores | 12 | 10 |
| Memory | 24 GB | 24 GB |
| Theoretical Bandwidth | ~200 GB/s | ~273 GB/s |

---

## GEMM Performance (Float16)

### PyTorch MPS

| Matrix Size | M5 | M4 Pro | M5/M4 Ratio |
|-------------|-----|--------|-------------|
| 1024×1024 | 6.32 TFLOPS | 2.78 TFLOPS | **2.3x** |
| 2048×2048 | 12.62 TFLOPS | 5.64 TFLOPS | **2.2x** |
| 4096×4096 | 13.84 TFLOPS | 6.03 TFLOPS | **2.3x** |
| 8192×8192 | 12.49 TFLOPS | 6.07 TFLOPS | **2.1x** |

### MLX

| Matrix Size | M5 | M4 Pro | M5/M4 Ratio |
|-------------|-----|--------|-------------|
| 1024×1024 | 2.67 TFLOPS | 3.94 TFLOPS | 0.7x |
| 2048×2048 | 3.59 TFLOPS | 5.67 TFLOPS | 0.6x |
| 4096×4096 | 3.62 TFLOPS | 6.04 TFLOPS | 0.6x |
| 8192×8192 | — | 6.08 TFLOPS | — |

**Key Finding:** PyTorch MPS on M5 is excellent (13.8 TFLOPS). MLX on M5 shows lower performance — likely needs MLX update for M5 chip.

---

## Memory Bandwidth

| Metric | M5 | M4 Pro | M4/M5 Ratio |
|--------|-----|--------|-------------|
| Peak Read | 118 GB/s | 253 GB/s | **2.1x** |
| Peak Copy | 119 GB/s | 230 GB/s | **1.9x** |
| Utilization | 59% | 93% | — |

**Key Finding:** M4 Pro achieves 2x higher memory bandwidth.

---

## Transformer Layer Performance (Float16)

### Forward Pass

| Seq Length | M5 | M4 Pro | M5/M4 Ratio |
|------------|-----|--------|-------------|
| 256 | 9.11 ms | 23.97 ms | **2.6x** |
| 512 | 18.22 ms | 46.80 ms | **2.6x** |
| 1024 | 41.61 ms | 96.02 ms | **2.3x** |

### Backward Pass

| Seq Length | M5 | M4 Pro | M5/M4 Ratio |
|------------|-----|--------|-------------|
| 256 | 18.25 ms | 89.48 ms | **4.9x** |
| 512 | 39.52 ms | 94.98 ms | **2.4x** |
| 1024 | 86.88 ms | 187.17 ms | **2.2x** |

### Full Training Step (Forward + Backward)

| Seq Length | M5 | M4 Pro | M5/M4 Ratio |
|------------|-----|--------|-------------|
| 256 | 27.36 ms | 113.45 ms | **4.1x** |
| 512 | 57.74 ms | 141.78 ms | **2.5x** |
| 1024 | 128.49 ms | 283.19 ms | **2.2x** |

---

## Attention Performance (Float16)

| Seq Length | M5 | M4 Pro | Ratio |
|------------|-----|--------|-------|
| 512 | 2.34 TFLOPS | 2.45 TFLOPS | ~Same |
| 1024 | 2.79 TFLOPS | 2.92 TFLOPS | ~Same |
| 2048 | 2.94 TFLOPS | 3.06 TFLOPS | ~Same |
| 4096 | 2.85 TFLOPS | 3.09 TFLOPS | ~Same |

**Key Finding:** Attention is memory-bound — similar on both machines.

---

## Fine-tuning (Float16, PyTorch MPS)

| Metric | M5 | M4 Pro |
|--------|-----|--------|
| Model | Mistral-7B | Mistral-7B |
| Precision | float16 | float16 |
| Tokens/sec | **174** | **104.5** |
| Peak Memory | 15.3 GB | 21.0 GB |

**M5 is 1.7x faster for fine-tuning.**

---

## Summary Table

| Metric | M5 | M4 Pro | Winner |
|--------|-----|--------|--------|
| **Peak GEMM (FP16)** | 13.84 TFLOPS | 6.07 TFLOPS | **M5 2.3x** |
| **Memory Bandwidth** | 118 GB/s | 253 GB/s | **M4 2.1x** |
| **Training Step** | 57.74 ms | 141.78 ms | **M5 2.5x** |
| **Fine-tune tok/s** | 174 | 104.5 | **M5 1.7x** |
| **MFU (FP16)** | 58.9% | 30.4% | **M5** |

---

## Distributed Strategy

### For Training (Compute-Bound)
- **M5 should do more work** — 2.3x faster compute
- Split: 70% M5, 30% M4 Pro

### For Inference (Memory-Bound)  
- **M4 Pro has advantage** — 2x faster memory bandwidth
- M4 Pro handles attention/KV cache, M5 handles FFN

### Thunderbolt Traffic Expectations
| Strategy | Pattern | Bottleneck |
|----------|---------|------------|
| Tensor Parallelism | Constant 3-4 GB/s | Interconnect |
| Expert Parallelism | Bursty 0-5 GB/s | Compute on M4 |

---

## Raw Data Locations

```
results/raw/M5/
├── system_info.json
├── pytorch_mps_benchmarks.json
├── layer_benchmark_float16.json
├── memory_benchmarks.json
├── mfu_utilization.json
└── pytorch_mps_mistral_finetune.json

results/raw/M4_Pro/
├── system_info.json
├── pytorch_mps_benchmarks.json
├── mlx_benchmarks.json
├── layer_benchmark_float16.json
├── memory_benchmarks.json
├── mfu_utilization.json
├── mlx_lora_finetuning.json
└── pytorch_mps_finetuning.json
```
