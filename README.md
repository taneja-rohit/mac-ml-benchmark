# Mac ML Benchmark Suite 🍎

Benchmark ML performance on Apple Silicon. Tested on M5 (24GB), ready for M4 Pro.

## Quick Start (M4 Pro)

```bash
git clone https://github.com/taneja-rohit/mac-ml-benchmark.git
cd mac-ml-benchmark
python3 -m venv venv && source venv/bin/activate
pip install torch transformers accelerate mlx mlx-lm datasets llama-cpp-python

# Run all benchmarks (results auto-save to results/raw/M4_Pro/)
python run_benchmarks.py
```

---

## Results: Apple M5 (24GB)

### Key Numbers

| Metric | PyTorch+MPS | MLX (4-bit) | llama.cpp |
|--------|-------------|-------------|-----------|
| **GEMM (FP16)** | **13.8 TFLOPS** | 3.6 TFLOPS | N/A |
| **Inference** | 7.7 t/s | **26.5 t/s** | 24.0 t/s |
| **Training** | **174 t/s** | 130 t/s | N/A |
| **Memory** | 14.5 GB | 4.5 GB | 5.0 GB |

### MFU (Utilization)

| Metric | Achieved | Theoretical | MFU |
|--------|----------|-------------|-----|
| FP16 GEMM | 14.1 TFLOPS | 24 TFLOPS | **59%** |
| Memory BW | 118 GB/s | 200 GB/s | **59%** |
| Model Training | 7.6 TFLOPS | 24 TFLOPS | **32%** |

### Layer-by-Layer (float16 vs float16)

| Component | PyTorch+MPS | MLX | Winner |
|-----------|-------------|-----|--------|
| Attention | 2.5ms | 6.4ms | **PyTorch 2.6x** |
| FFN | 6.6ms | 25.1ms | **PyTorch 3.8x** |
| Backward | 18.3ms | 51.9ms | **PyTorch 2.8x** |

---

## What We Learned

1. **PyTorch+MPS CAN run Mistral-7B** — The 12GB limit is per-tensor, not total memory
2. **At same precision, PyTorch is 2.5-3.5x faster** than MLX
3. **MLX wins for inference** only because of 4-bit quantization (less memory bandwidth)
4. **59% MFU is excellent** — same as NVIDIA datacenter GPUs on real workloads

---

## Files

```
results/raw/M5/
├── system_info.json              # Hardware specs
├── pytorch_mps_benchmarks.json   # GEMM/Attention benchmarks
├── layer_benchmark_float16.json  # Fair PyTorch vs MLX comparison
├── framework_comparison.json     # All 3 frameworks
├── mfu_utilization.json          # MFU analysis
├── memory_benchmarks.json        # Bandwidth tests
└── pytorch_mps_mistral_finetune.json  # Training benchmark
```

---

## Documentation

- **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)** — Detailed analysis with explanations
- **[CONSTRAINTS_AND_LEARNINGS.md](CONSTRAINTS_AND_LEARNINGS.md)** — Technical deep-dive

---

## License

MIT
