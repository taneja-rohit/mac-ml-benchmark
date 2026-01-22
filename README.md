# Mac ML Benchmark Suite 🍎

Comprehensive ML benchmarks for Apple Silicon Macs (M5/M4 Pro).  
Compares **PyTorch + MPS** vs **MLX** for deep learning workloads.

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run all benchmarks
python run_benchmarks.py

# Or individual phases
python run_benchmarks.py --discovery-only   # Hardware info
python run_benchmarks.py --pytorch-only     # PyTorch + MPS
python run_benchmarks.py --mlx-only         # MLX
python run_benchmarks.py --memory-only      # Memory bandwidth
```

## Results (Apple M5 - 24GB)

### Compute Performance

| Benchmark | PyTorch+MPS | MLX |
|-----------|-------------|-----|
| GEMM float32 (2048) | 3.61 TFLOPS | 3.33 TFLOPS |
| GEMM float16 (4096) | **13.39 TFLOPS** 🏆 | 3.59 TFLOPS |
| Attention (seq=2048) | 2.94 TFLOPS | 2.42 TFLOPS |

**Key Finding**: PyTorch float16 is **3.7× faster** than MLX!

### Memory Bandwidth

| Operation | Bandwidth |
|-----------|-----------|
| Read | 113 GB/s |
| Write | 113 GB/s |
| Copy (R+W) | **119 GB/s** |

Apple M5 theoretical: ~200 GB/s (we achieve ~60% utilization)

### GPU Memory Limits

| Test | Result |
|------|--------|
| Max single allocation | 12 GB |
| Total unified memory | 24 GB |

## Project Structure

```
mac-ml-benchmark/
├── run_benchmarks.py          # Main orchestrator
├── discovery/
│   └── system_info.py         # Hardware detection
├── benchmarks/
│   ├── compute/
│   │   ├── pytorch_mps.py     # PyTorch+MPS benchmarks
│   │   └── mlx_benchmarks.py  # MLX benchmarks
│   └── memory/
│       └── bandwidth.py       # Memory bandwidth tests
├── results/
│   ├── raw/                   # JSON data
│   └── reports/               # Summary reports
└── config/
    └── benchmark_config.yaml
```

## Why Can't We Run Mistral-7B?

**Short answer**: MPS can only allocate 12GB per tensor, and Mistral-7B needs ~14GB.

```
Mistral-7B memory breakdown:
├── Model weights (fp16): ~14 GB
├── KV cache (seq=2048):  ~2-4 GB
├── Activations:          ~2-4 GB
└── Total needed:         ~18-22 GB

But MPS max allocation = 12 GB ❌
```

**Workarounds**:
1. Use 4-bit quantized model (MLX supports this)
2. Use CPU offloading
3. Use smaller models (Mistral-1B, Phi-2)

## License

MIT
