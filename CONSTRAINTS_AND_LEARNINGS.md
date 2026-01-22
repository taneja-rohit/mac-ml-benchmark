# Constraints, Learnings & The Bitter Truth About Apple Silicon ML

> "The ships hung in the sky in much the same way that bricks don't."
> — Douglas Adams
>
> Apple Silicon handles ML workloads in much the same way.

---

## Table of Contents
1. [The 12GB Wall: MPS Tensor Limits](#the-12gb-wall)
2. [Why Mistral-7B Won't Fit (Math Time)](#why-mistral-7b-wont-fit)
3. [PyTorch MPS vs MLX: WTF is the Difference?](#pytorch-mps-vs-mlx)
4. [Quantization: The Industry's Escape Hatch](#quantization)
5. [Why NVIDIA is Winning](#why-nvidia-is-winning)
6. [Quality vs. Compression Tradeoffs](#quality-tradeoffs)
7. [Apple's Missed Opportunities](#apples-missed-opportunities)
8. [Practical Recommendations](#practical-recommendations)

---

## The 12GB Wall: MPS Tensor Limits {#the-12gb-wall}

### What's Actually Happening

Apple's Metal Performance Shaders (MPS) — the backend that makes PyTorch run on Apple GPUs — has a dirty little secret: **it uses 32-bit signed integers for tensor dimension indexing**.

```
Maximum elements per dimension = 2^31 - 1 = 2,147,483,647

For a contiguous float16 tensor:
  Max size ≈ 2^31 × 2 bytes = 4 GB per dimension
  
For float32:
  Max size ≈ 2^31 × 4 bytes = 8 GB per dimension

In practice, with overhead: ~12 GB max allocation observed
```

### The Infuriating Part

Your MacBook has **24 GB of unified memory**. The GPU and CPU share it. In theory, you could load a 20GB model. In practice, MPS says "nah."

This is a **software limitation**, not hardware. Apple's Metal team optimized for:
- Video editing (many small textures)
- Image processing (bounded dimensions)
- NOT giant weight matrices sitting in VRAM

```
╔═══════════════════════════════════════════════════════════════════╗
║  YOUR MACBOOK'S EXISTENTIAL CRISIS                                ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   Physical RAM:        24 GB    "I have so much potential!"       ║
║   Usable for ML:       12 GB    "But MPS won't let me use it"    ║
║   Mistral-7B needs:    14 GB    "So close, yet so far"           ║
║                                                                   ║
║   Status: PAIN                                                    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### Technical Root Cause

```c
// Somewhere in Apple's Metal framework (conceptually):
struct MPSNDArrayDescriptor {
    int32_t dimensions[8];    // <-- HERE'S THE PROBLEM
    // ...
};

// When you try to allocate > INT_MAX elements:
// "NDArray dimension length > INT_MAX" 💀
```

NVIDIA's CUDA uses `size_t` (64-bit) for this. Apple chose `int32_t`. Presumably in 2016 when 8GB GPUs were exotic.

---

## Why Mistral-7B Won't Fit {#why-mistral-7b-wont-fit}

Let's do the math. Mistral-7B has 7.24 billion parameters.

### Memory Breakdown (float16 inference)

```
MISTRAL-7B ARCHITECTURE:
─────────────────────────────────────────────────────────
Layers:           32 transformer blocks
Hidden dim:       4096
Intermediate:     14336 (FFN)
Vocab size:       32000
Attention heads:  32
KV heads:         8 (Grouped Query Attention)

WEIGHT SIZES (float16 = 2 bytes per param):
─────────────────────────────────────────────────────────
Embedding:        32000 × 4096 × 2 =    256 MB
Per-layer:
  ├─ Q proj:      4096 × 4096 × 2 =      32 MB
  ├─ K proj:      4096 × 1024 × 2 =       8 MB  (GQA)
  ├─ V proj:      4096 × 1024 × 2 =       8 MB  (GQA)
  ├─ O proj:      4096 × 4096 × 2 =      32 MB
  ├─ Gate proj:   4096 × 14336 × 2 =    112 MB
  ├─ Up proj:     4096 × 14336 × 2 =    112 MB
  └─ Down proj:   14336 × 4096 × 2 =    112 MB
  Layer total:                          416 MB × 32 = 13.3 GB

LM Head:          4096 × 32000 × 2 =    256 MB

TOTAL WEIGHTS:    ~14 GB
─────────────────────────────────────────────────────────
```

### Runtime Memory

```
RUNTIME MEMORY (inference, seq_len=2048):
─────────────────────────────────────────────────────────
Weights:                               14.0 GB
KV Cache (32 layers × 2048 tokens):     2.0 GB
Activations (batch=1):                  0.5 GB
Framework overhead:                     0.5 GB
─────────────────────────────────────────────────────────
TOTAL:                                ~17.0 GB

MPS LIMIT:                             12.0 GB

VERDICT:                               ❌ DOESN'T FIT
```

---

## PyTorch MPS vs MLX: WTF is the Difference? {#pytorch-mps-vs-mlx}

This is the question everyone has. Let me break it down:

### The Stack Diagram

```
YOUR PYTHON CODE
      │
      ├─────────────────────────────────────────────────────┐
      │                                                     │
      ▼                                                     ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│      PyTorch            │                    │         MLX             │
│   (Meta/Facebook)       │                    │       (Apple)           │
│                         │                    │                         │
│  - 10+ years mature     │                    │  - Released Dec 2023    │
│  - Massive ecosystem    │                    │  - Apple-native         │
│  - CUDA-first design    │                    │  - NumPy-like API       │
└───────────┬─────────────┘                    └───────────┬─────────────┘
            │                                              │
            ▼                                              ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│     MPS Backend         │                    │    MLX Metal Backend    │
│  (Apple contribution    │                    │    (Apple internal)     │
│   to PyTorch)           │                    │                         │
│                         │                    │  - Lazy evaluation      │
│  - Bolted-on adapter    │                    │  - Unified memory aware │
│  - Translates CUDA ops  │                    │  - Apple-optimized      │
│    to Metal shaders     │                    │                         │
└───────────┬─────────────┘                    └───────────┬─────────────┘
            │                                              │
            └──────────────────┬───────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │    Apple Metal API      │
                    │    (GPU driver)         │
                    └───────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │   Apple Silicon GPU     │
                    │   (M5 - 12 cores)       │
                    └─────────────────────────┘
```

### Key Differences

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    PyTorch + MPS vs MLX                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Aspect              │ PyTorch + MPS          │ MLX                        ║
╠═════════════════════╪════════════════════════╪════════════════════════════╣
║ Who maintains it    │ Meta + Apple contribs  │ Apple ML Research          ║
║ Maturity            │ 10+ years (MPS: 2022)  │ ~1 year (Dec 2023)         ║
║ Design philosophy   │ CUDA-first, MPS bolted │ Apple-native from scratch  ║
║ Execution model     │ Eager (immediate)      │ Lazy (deferred)            ║
║ Memory model        │ CUDA-style (explicit)  │ Unified memory aware       ║
║ API                 │ PyTorch (industry std) │ NumPy-like (simpler)       ║
║ Ecosystem           │ Massive (HuggingFace)  │ Growing (mlx-lm, etc)      ║
║ float16 performance │ GREAT (13 TFLOPS)      │ Meh (3.5 TFLOPS)           ║
║ Quantization        │ bitsandbytes, GPTQ     │ Native 4-bit               ║
║ Training support    │ Full                   │ Full (LoRA friendly)       ║
║ Documentation       │ Excellent              │ Decent                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Why PyTorch float16 is 3.7x Faster Than MLX

This is the shocking finding from our benchmarks:

```
GEMM 4096x4096:
  PyTorch MPS float16:  13.4 TFLOPS  🏆
  MLX float16:           3.6 TFLOPS  
  
WHY?

1. PyTorch MPS uses MPSGraph
   - Apple's high-level ML graph API
   - Has hand-tuned GEMM kernels for float16
   - Benefits from years of Metal optimization

2. MLX uses raw Metal compute shaders
   - More flexible (lazy eval)
   - But less optimized for peak throughput
   - Still catching up on kernel optimization

3. The MPS float16 path hits Apple's "fast path"
   - Likely uses AMX (Apple Matrix coprocessor)
   - MLX may not be triggering this yet
```

### Where is Major Effort Going?

```
DEVELOPMENT INVESTMENT:

PyTorch (Meta):
├─ Main focus: CUDA, ROCm (AMD), XLA (TPU)
├─ MPS: ~5% of effort, mostly Apple contributions
└─ Future: torch.compile, inductor backend
   
MLX (Apple):
├─ Main focus: Apple Silicon optimization
├─ Growing fast: mlx-lm, mlx-vlm, mlx-audio
└─ Future: Unknown (Apple doesn't share roadmaps)

INDUSTRY MOMENTUM:
─────────────────────────────────────────────────────────────
Framework        │ GitHub Stars │ Contributors │ HF Models
─────────────────┼──────────────┼──────────────┼───────────
PyTorch          │ 85k          │ 3,500+       │ 500k+
MLX              │ 18k          │ 100+         │ 1,000+
─────────────────────────────────────────────────────────────

VERDICT: 
- PyTorch has massive momentum, MPS is an afterthought
- MLX is Apple's bet, growing but niche
- For production: PyTorch (ecosystem)
- For Mac-specific: MLX (if you can port)
```

### The Uncomfortable Truth

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         THE REAL SITUATION                                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   Meta (PyTorch):  "MPS? Sure, Apple can maintain that."                 ║
║   Apple (MPS):     "We'll do the minimum to not embarrass ourselves."    ║
║   Apple (MLX):     "Here's our REAL answer for Apple Silicon."           ║
║   ML Community:    "We have 10 million lines of PyTorch code."           ║
║   Apple (MLX):     "Cool, rewrite it."                                   ║
║   ML Community:    "..."                                                  ║
║                                                                           ║
║   Result: Everyone uses PyTorch+MPS and complains about it.              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Quantization: The Industry's Escape Hatch {#quantization}

### What is Quantization?

Converting weights from high-precision (float16/32) to low-precision (int8/int4):

```
FLOAT16 (16 bits per weight):
┌────────────────────────────────────────────────┐
│ S │ EEEEE │ MMMMMMMMMM │   Range: ±65504      │
│ 1 │   5   │     10     │   Precision: ~0.001  │
└────────────────────────────────────────────────┘

INT8 (8 bits per weight):
┌────────────────────────────┐
│ SSSSSSSS │  Range: -128 to 127              │
│    8     │  + scale factor per group        │
└────────────────────────────┘

INT4 (4 bits per weight):
┌────────────┐
│ SSSS │  Range: -8 to 7                      │
│  4   │  + scale + zero-point per group      │
└────────────┘
```

### Memory Savings

```
MISTRAL-7B MEMORY BY PRECISION:
─────────────────────────────────────────────────
Precision    Size        Fits MPS?    Quality
─────────────────────────────────────────────────
float32      28 GB       ❌ No        100%
float16      14 GB       ❌ No        ~100%
int8          7 GB       ✅ Yes       ~99%
int4 (GPTQ)   4 GB       ✅ Yes       ~97%
int4 (AWQ)    4 GB       ✅ Yes       ~98%
─────────────────────────────────────────────────
```

### Industry-Standard Methods

| Method | How it Works | Pros | Cons |
|--------|-------------|------|------|
| **GPTQ** | Calibration-based 4-bit, Hessian-weighted | Fast inference, wide support | Needs calibration data |
| **AWQ** | Activation-aware, protects salient weights | Better quality than GPTQ | Slightly more complex |
| **bitsandbytes** | On-the-fly int8/int4 | Easy integration | Slower than pre-quantized |
| **GGUF** | CPU+GPU hybrid, mixed precision | Runs anywhere | Not optimal for pure GPU |
| **MLX Native** | Apple-optimized 4-bit | Best on Apple Silicon | Apple-only |

---

## Why NVIDIA is Winning {#why-nvidia-is-winning}

### The Technical Gap

```
╔══════════════════════════════════════════════════════════════════════╗
║                    NVIDIA vs APPLE SILICON                           ║
╠══════════════════════════════════════════════════════════════════════╣
║ Metric              │ NVIDIA H100      │ Apple M5                    ║
╠═════════════════════╪══════════════════╪═════════════════════════════╣
║ FP16 TFLOPS         │ 1,979            │ ~14 (measured)              ║
║ Memory              │ 80 GB HBM3       │ 24 GB unified               ║
║ Memory BW           │ 3.35 TB/s        │ ~120 GB/s (measured)        ║
║ Tensor Cores        │ Yes (4th gen)    │ No                          ║
║ Max tensor size     │ 80 GB            │ 12 GB (MPS limit)           ║
║ Flash Attention     │ Native           │ Hacky/limited               ║
║ CUDA ecosystem      │ Massive          │ N/A                         ║
║ Price               │ $30,000          │ $2,499                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### The Ecosystem Moat

```
NVIDIA's unfair advantages:

1. CUDA (2007) - 17 years of momentum
   └─ Every ML framework: "CUDA first, maybe others later"

2. cuDNN - Hand-tuned kernels for every operation
   └─ Apple's MPS: "Here's some generic Metal shaders, good luck"

3. Tensor Cores - Hardware matrix multiply units
   └─ M5 GPU: General-purpose ALUs doing matrix math

4. NVLink/NVSwitch - Multi-GPU at 900 GB/s
   └─ Apple: "Thunderbolt 4 at 40 Gbps, take it or leave it"

5. Software stack depth:
   NVIDIA: CUDA → cuDNN → cuBLAS → TensorRT → Triton → vLLM
   Apple:  Metal → MPS → ... → ... → "it works on MacBooks I guess"
```

---

## Quality vs. Compression Tradeoffs {#quality-tradeoffs}

### Benchmark Reality

```
MISTRAL-7B QUALITY BY QUANTIZATION:
─────────────────────────────────────────────────────────────────────
Benchmark        │ FP16    │ INT8    │ GPTQ-4  │ AWQ-4  
─────────────────┼─────────┼─────────┼─────────┼─────────
MMLU (knowledge) │ 60.1%   │ 59.8%   │ 59.2%   │ 59.5%  
HellaSwag        │ 81.3%   │ 81.2%   │ 80.8%   │ 81.0%  
HumanEval (code) │ 32.0%   │ 31.5%   │ 30.5%   │ 31.2%  
GSM8K (math)     │ 52.2%   │ 51.5%   │ 49.8%   │ 50.9%  
─────────────────┴─────────┴─────────┴─────────┴─────────

Key insight: Knowledge/reasoning barely affected (-0.5%)
             Math/code takes the hit (-3-5%)
```

---

## Apple's Missed Opportunities {#apples-missed-opportunities}

### What Apple Got Right ✅

- **Unified Memory Architecture** - Could enable massive models on laptops
- **Power Efficiency** - 30W vs 700W for comparable tasks
- **MLX Framework** - Lazy evaluation, NumPy-like API
- **Hardware potential** - Neural Engine exists (16 TOPS)

### What Apple Got Wrong ❌

- **MPS INT_MAX Limitation** - Inexcusable in 2024
- **No Tensor Cores** - GPU does generic ALU math
- **Neural Engine Locked Down** - 16 TOPS sitting unused, only via CoreML
- **Half-baked Flash Attention** - MPS implementation incomplete
- **Ecosystem Neglect** - PyTorch MPS is community-maintained mostly

---

## Practical Recommendations {#practical-recommendations}

### For Your M5 MacBook (24GB)

```
MODEL SELECTION GUIDE:
─────────────────────────────────────────────────────────────
Model Size (params)  │ Precision │ Memory   │ Verdict
─────────────────────┼───────────┼──────────┼─────────────────
< 3B (Phi-2, etc)    │ FP16      │ ~6 GB    │ ✅ Runs great
7B (Mistral, Llama)  │ INT4      │ ~4 GB    │ ✅ Use quantized
7B                   │ FP16      │ ~14 GB   │ ❌ Won't fit
13B                  │ INT4      │ ~7 GB    │ ✅ Works
70B                  │ Any       │ 35+ GB   │ ❌ Forget it
─────────────────────────────────────────────────────────────
```

### Commands to Run

```bash
# MLX 4-bit inference (easiest, best for Mac)
pip install mlx-lm
mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
                --prompt "Your prompt here"

# PyTorch INT8 via bitsandbytes
pip install bitsandbytes accelerate
# Load with: load_in_8bit=True

# PyTorch GPTQ-4
pip install auto-gptq optimum
# Load TheBloke's GPTQ models from HuggingFace
```

---

## Benchmark Results from This Machine

```
APPLE M5 (24GB) - MEASURED PERFORMANCE:
═══════════════════════════════════════════════════════════════
COMPUTE:
  PyTorch+MPS GEMM (float16):    13.4 TFLOPS (peak)
  PyTorch+MPS GEMM (float32):     3.6 TFLOPS
  MLX GEMM (all precisions):      3.5 TFLOPS
  Attention (seq=2048):           2.9 TFLOPS

MEMORY:
  Bandwidth (copy):             119 GB/s
  Max single allocation:         12 GB

FRAMEWORK WINNER:
  PyTorch+MPS float16 beats MLX by 3.7x on GEMM
═══════════════════════════════════════════════════════════════
```

---

*"The answer to the ultimate question of ML, the universe, and everything is: use 4-bit quantization and stop complaining about MPS."*

— Generated 2026-01-22, Apple M5 (24GB)
