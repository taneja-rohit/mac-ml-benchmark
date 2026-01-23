# Mac ML Benchmark Suite 🍎

Comprehensive ML benchmarks for Apple Silicon Macs.  
Compares **PyTorch+MPS** vs **MLX** vs **llama.cpp** for deep learning workloads.

## Key Findings (Apple M5, 24GB)

| Framework | Inference | Training | Memory | Best For |
|-----------|-----------|----------|--------|----------|
| **PyTorch+MPS** | 7.7 t/s | **174 t/s** 🏆 | 14.5 GB | Training |
| **MLX** (4-bit) | **26.5 t/s** 🏆 | 130 t/s | 4.5 GB | Inference |
| **llama.cpp** | 24.0 t/s | N/A | 5.0 GB | Deployment |

> **Note**: Earlier versions incorrectly stated PyTorch+MPS cannot run Mistral-7B.  
> This was WRONG. PyTorch CAN load and train Mistral-7B in float16.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/taneja-rohit/mac-ml-benchmark.git
cd mac-ml-benchmark
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install mlx mlx-lm datasets llama-cpp-python

# Run all benchmarks
python run_benchmarks.py

# Or run specific tests
python run_benchmarks.py --discovery-only   # Hardware info
python run_benchmarks.py --pytorch-only     # PyTorch+MPS
python run_benchmarks.py --mlx-only         # MLX
```

## GEMM Performance (Matrix Multiply)

| Size | PyTorch float32 | PyTorch float16 | MLX |
|------|-----------------|-----------------|-----|
| 2048 | 3.43 TFLOPS | 12.62 TFLOPS | 3.32 TFLOPS |
| 4096 | 3.16 TFLOPS | **13.84 TFLOPS** 🏆 | 3.25 TFLOPS |

**PyTorch float16 is 3.8x faster than MLX** due to Apple's hand-tuned MPSGraph kernels.

## Memory Bandwidth

| Metric | Value |
|--------|-------|
| Read | 113 GB/s |
| Write | 113 GB/s |
| Copy | **119 GB/s** |
| Theoretical | ~200 GB/s |
| Utilization | ~60% |

## The MPS "12GB Limit" (Corrected)

**The limit is PER-TENSOR, not total memory.**

```
Mistral-7B:
├── Total size: 14.48 GB
├── Largest tensor: ~1.1 GB
└── Result: ✅ LOADS SUCCESSFULLY

The model is distributed across many tensors,
each under 12GB, so it fits.
```

## Project Structure

```
mac-ml-benchmark/
├── run_benchmarks.py              # Main runner
├── BENCHMARK_RESULTS.md           # Detailed results & analysis
├── CONSTRAINTS_AND_LEARNINGS.md   # Technical deep-dive
├── discovery/
│   └── system_info.py             # Hardware detection
├── benchmarks/
│   ├── compute/                   # GEMM, Attention benchmarks
│   └── memory/                    # Bandwidth tests
├── data/                          # Alpaca dataset (for training)
└── results/
    └── raw/
        └── M5/                    # Your machine's results
            └── M4_Pro/            # (When you run on other machine)
```

## Fine-Tuning Results

Both PyTorch+MPS and MLX can fine-tune Mistral-7B:

```bash
# MLX (easier, lower memory)
python -m mlx_lm lora --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
    --data ./data --train --iters 50

# PyTorch (faster training throughput)
# See benchmarks/compute/pytorch_mps.py
```

| Framework | Method | Speed | Peak Memory |
|-----------|--------|-------|-------------|
| **PyTorch+MPS** | LM Head only | **174 t/s** | 15.3 GB |
| **MLX** | LoRA | 130 t/s | 7.5 GB |

## Running on M4 Pro

Results are automatically saved to `results/raw/M4_Pro/`:

```bash
git clone https://github.com/taneja-rohit/mac-ml-benchmark.git
cd mac-ml-benchmark
python run_benchmarks.py  # Detects M4 Pro and saves separately
```

## License

MIT
