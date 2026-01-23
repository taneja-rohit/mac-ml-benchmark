# Silicon ML Benchmarks: 
This repository is a technical investigation into the evolving architecture for GenAI. It moves beyond "speed tests" to analyze **Arithmetic Intensity**, **Model FLOPs Utilization (MFU)**, and the **Inference Paradox** .

---

##  Summary: The "Inference Paradox"

Machines being used: 
*   **M5 (The Compute Monster):** Optimized for **Compute Density**. It introduces integrated **Neural Accelerators** into every GPU core. Result: **2.5x faster training** and fine-tuning than the M4 Pro.
*   **M4 Pro (The Bandwidth Beast):** Optimized for **Data Throughput**. Its wider memory bus achieves 2x higher bandwidth. Result: **2.1x faster inference** (token generation) than the M5.

**The TL;DR:** For training/fine-tuning where math is the bottleneck, the M5 is king. For inference where loading weights is the bottleneck, the M4 Pro remains the king.

---

## 🛠️ Getting Started

### 1. Requirements
*   macOS 15.0+
*   Python 3.9+
*   Thunderbolt 4 cable (for Phase 2 Distributed benchmarks)

### 2. Setup
```bash
git clone https://github.com/taneja-rohit/mac-ml-benchmark.git
cd mac-ml-benchmark
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Benchmarks
```bash
# Results auto-save to results/raw/<CHIP_NAME>/
python3 run_benchmarks.py
```

---

## The "Why": Methodology & Reasoning

### Why Benchmark Layers Separately?
Modern LLM performance is often obscured by framework overhead. We break benchmarks down into:
1.  **Raw GEMM**: To measure peak theoretical compute (TFLOPS).
2.  **Attention Mechanisms**: To measure memory-latency sensitivity.
3.  **Transformer Blocks**: To measure fused-kernel efficiency.
4.  **End-to-End (Mistral-7B)**: To measure real-world application performance.

### The Dance of Bandwidth and Compute Density
Models move from fine-tuning to recall to reasoning, the bottleneck takes a pendulamic shigt from **loading weights** to **calculating gradients**. Interestingly, Apple's latest M5 is prioritizing "FLOPs per Watt" over "Bytes per Second.". 

---

## 💻 Hardware Analysis: M5 vs. M4 Pro

| Feature | Apple M5 (Base) | Apple M4 Pro |
| :--- | :--- | :--- |
| **Primary Strength** | **Compute Density** | **Memory Throughput** |
| **New Tech** | Integrated Neural Accelerators | High-Width Memory Bus |
| **Peak FP16 TFLOPS** | **13.8 TFLOPS** | 6.1 TFLOPS |
| **Peak Bandwidth** | 118 GB/s | **253 GB/s** |
| **Ridge Point** | 116.9 FLOPs/Byte | 24.0 FLOPs/Byte |

### Reasoning:
*   **M5 wins Training** because its GPU cores now contain dedicated matrix-math units (similar to NVIDIA Tensor Cores), allowing it to process backward passes significantly faster.
*   **M4 Pro wins Inference** because token generation is a "weight-reading" game. The M4 Pro's 256-bit bus (approx) moves weights twice as fast as the M5's restricted 128-bit bus.

---

## ⚡ Framework Battle: PyTorch+MPS vs. MLX

We tested both frameworks at the **exact same precision (Float16)** to remove quantization bias.

*   **PyTorch + MPS (The Powerhouse):** Currently achieves **2.3x higher TFLOPS** on M5. Apple’s `MPSGraph` is highly optimized for the new M5 Neural Accelerators.
*   **MLX (The Efficient):** Excellent for memory-constrained inference and 4-bit quantization, but currently underperforms in raw FP16 GEMM on M5, suggesting a need for kernel updates for the 2026 hardware.

---

## 📊 Visual Proof: Roofline Analysis

The **Roofline Model** below proves that Mistral-7B inference is strictly **Memory-Bound**.

![Roofline Comparison](results/reports/roofline_comparison.png)

*   **Memory Bound Zone:** Both chips are stuck on the "slanted roof." The M4 Pro's roof is higher, thus it runs faster.
*   **Compute Bound Zone:** During training, we hit the "flat top." The M5's top is much higher, thus it trains faster.

---

##  Raw Data

### 1. GEMM Performance (Float16)
| Matrix Size | M5 (TFLOPS) | M4 Pro (TFLOPS) |
| :--- | :--- | :--- |
| 1024x1024 | 6.32 | 2.78 |
| 4096x4096 | **13.84** | 6.03 |
| 8192x8192 | **12.49** | 6.07 |

### 2. Memory Bandwidth (Achieved)
| Metric | M5 | M4 Pro |
| :--- | :--- | :--- |
| Sequential Read | 118.4 GB/s | **253.2 GB/s** |
| Sequential Write | 110.2 GB/s | **170.5 GB/s** |
| Memory Copy | 118.9 GB/s | **230.3 GB/s** |

### 3. Training Step (Mistral-7B Layer, seq=512)
| Phase | M5 | M4 Pro |
| :--- | :--- | :--- |
| Forward Pass | **18.2 ms** | 46.8 ms |
| Backward Pass | **39.5 ms** | 95.0 ms |
| **Total Step** | **57.7 ms** | **141.8 ms** |

---

##  Repository Structure
*   `benchmarks/`: Core logic for compute, memory, and model tests.
*   `distributed/`: Tools for multi-Mac training (Traffic monitoring).
*   `visualizations/`: Roofline and traffic plotting scripts.
*   `results/raw/`: JSON artifacts from every run.

---

##  Further Reading
- [Detailed Analysis & Strategy](M5_VS_M4_PRO_ANALYSIS.md)
- [Technical Constraints & Learnings](CONSTRAINTS_AND_LEARNINGS.md)

## Next: What am I thinking 
- Deepseek MoE 16B with distributed fine-tune over M5 and M4 connected over thunderbolt
- Disaggregated fine-tuning (Prefill on flops heavy M5 and decode on high throughput M4) 


