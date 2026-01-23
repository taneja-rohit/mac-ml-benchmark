"""
MLX (Apple's ML Framework) Compute Benchmarks

MLX is designed specifically for Apple Silicon with:
- Lazy evaluation
- Unified memory
- Efficient GPU utilization
"""

import time
import json
import gc
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("WARNING: MLX not installed. Install with: pip install mlx mlx-lm")

@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    framework: str = "mlx"
    size: str = ""
    dtype: str = ""
    time_ms: float = 0.0
    tflops: float = 0.0
    bandwidth_gbs: float = 0.0
    memory_mb: float = 0.0
    utilization_pct: float = 0.0
    extra: Dict[str, Any] = None
    
    def to_dict(self):
        return asdict(self)

def sync_mlx():
    """Synchronize MLX operations (evaluate lazy computation)."""
    if MLX_AVAILABLE:
        mx.eval(mx.zeros(1))  # Force sync

def clear_memory():
    """Clear memory."""
    gc.collect()

def calculate_gemm_flops(M: int, N: int, K: int) -> int:
    """Calculate FLOPS for matrix multiply."""
    return 2 * M * N * K

# ============================================================================
# GEMM BENCHMARKS
# ============================================================================

def benchmark_gemm_mlx(
    sizes: List[int] = [512, 1024, 2048, 4096, 8192],
    dtypes: List[str] = ["float32", "float16"],
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> List[BenchmarkResult]:
    """Benchmark matrix multiplication with MLX."""
    
    if not MLX_AVAILABLE:
        return [BenchmarkResult(name="gemm", extra={"error": "MLX not available"})]
    
    results = []
    
    print("\n" + "="*60)
    print("GEMM BENCHMARK (MLX)")
    print("="*60)
    print(f"Device: {mx.default_device()}")
    
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    
    for dtype_name in dtypes:
        if dtype_name not in dtype_map:
            continue
        dtype = dtype_map[dtype_name]
        
        for size in sizes:
            M, N, K = size, size, size
            
            try:
                # Allocate matrices
                A = mx.random.normal(shape=(M, K)).astype(dtype)
                B = mx.random.normal(shape=(K, N)).astype(dtype)
                mx.eval(A, B)  # Force materialization
                
                # Warmup
                for _ in range(warmup_iters):
                    C = A @ B
                    mx.eval(C)
                
                # Benchmark
                times = []
                for _ in range(bench_iters):
                    start = time.perf_counter()
                    C = A @ B
                    mx.eval(C)  # Force computation
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # Calculate metrics
                flops = calculate_gemm_flops(M, N, K)
                tflops = (flops / avg_time) / 1e9
                
                # Bandwidth (approximate)
                bytes_elem = 4 if dtype_name == "float32" else 2
                bytes_moved = bytes_elem * (M * K + K * N + M * N)
                bandwidth = (bytes_moved / avg_time) / 1e6
                
                result = BenchmarkResult(
                    name=f"gemm_{size}x{size}",
                    size=f"{size}x{size}",
                    dtype=dtype_name,
                    time_ms=avg_time,
                    tflops=tflops,
                    bandwidth_gbs=bandwidth,
                    extra={"std_ms": std_time, "iterations": bench_iters}
                )
                results.append(result)
                
                print(f"  GEMM {size:5d}x{size:5d} ({dtype_name:8s}): "
                      f"{avg_time:8.2f}ms  {tflops:6.2f} TFLOPS  {bandwidth:8.1f} GB/s")
                
                # Clean up
                del A, B, C
                clear_memory()
                
            except Exception as e:
                print(f"  GEMM {size}x{size} ({dtype_name}): FAILED - {e}")
                results.append(BenchmarkResult(
                    name=f"gemm_{size}x{size}",
                    size=f"{size}x{size}",
                    dtype=dtype_name,
                    extra={"error": str(e)}
                ))
    
    return results

# ============================================================================
# ATTENTION BENCHMARKS
# ============================================================================

def benchmark_attention_mlx(
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> List[BenchmarkResult]:
    """Benchmark self-attention with MLX."""
    
    if not MLX_AVAILABLE:
        return [BenchmarkResult(name="attention", extra={"error": "MLX not available"})]
    
    results = []
    
    print("\n" + "="*60)
    print("ATTENTION BENCHMARK (MLX)")
    print("="*60)
    
    for seq_len in seq_lengths:
        try:
            # Create Q, K, V
            Q = mx.random.normal(shape=(batch_size, num_heads, seq_len, head_dim)).astype(mx.float16)
            K = mx.random.normal(shape=(batch_size, num_heads, seq_len, head_dim)).astype(mx.float16)
            V = mx.random.normal(shape=(batch_size, num_heads, seq_len, head_dim)).astype(mx.float16)
            mx.eval(Q, K, V)
            
            scale = 1.0 / (head_dim ** 0.5)
            
            def attention_forward():
                scores = (Q @ mx.transpose(K, (0, 1, 3, 2))) * scale
                probs = mx.softmax(scores, axis=-1)
                output = probs @ V
                return output
            
            # Warmup
            for _ in range(warmup_iters):
                out = attention_forward()
                mx.eval(out)
            
            # Benchmark
            times = []
            for _ in range(bench_iters):
                start = time.perf_counter()
                out = attention_forward()
                mx.eval(out)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            
            # FLOPS
            qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
            pv_flops = 2 * batch_size * num_heads * seq_len * head_dim * seq_len
            total_flops = qk_flops + pv_flops
            tflops = (total_flops / avg_time) / 1e9
            
            # Memory for attention scores
            attn_scores_mb = (batch_size * num_heads * seq_len * seq_len * 2) / (1024**2)
            
            result = BenchmarkResult(
                name=f"attention_seq{seq_len}",
                size=f"seq={seq_len}, heads={num_heads}, dim={head_dim}",
                dtype="float16",
                time_ms=avg_time,
                tflops=tflops,
                memory_mb=attn_scores_mb,
                extra={
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "seq_len": seq_len,
                }
            )
            results.append(result)
            
            print(f"  Attention seq={seq_len:5d}: {avg_time:8.2f}ms  "
                  f"{tflops:6.2f} TFLOPS  attn_scores={attn_scores_mb:.1f}MB")
            
            del Q, K, V
            clear_memory()
            
        except Exception as e:
            print(f"  Attention seq={seq_len}: FAILED - {e}")
            results.append(BenchmarkResult(
                name=f"attention_seq{seq_len}",
                extra={"error": str(e)}
            ))
    
    return results

# ============================================================================
# TRANSFORMER BLOCK BENCHMARK
# ============================================================================

def benchmark_transformer_block_mlx(
    hidden_size: int = 4096,
    num_heads: int = 32,
    intermediate_size: int = 14336,
    seq_lengths: List[int] = [256, 512, 1024],
    batch_size: int = 1,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> List[BenchmarkResult]:
    """Benchmark a full transformer block with MLX."""
    
    if not MLX_AVAILABLE:
        return [BenchmarkResult(name="transformer_block", extra={"error": "MLX not available"})]
    
    results = []
    
    print("\n" + "="*60)
    print("TRANSFORMER BLOCK BENCHMARK (MLX)")
    print("="*60)
    
    for seq_len in seq_lengths:
        try:
            head_dim = hidden_size // num_heads
            
            # Weights
            W_qkv = mx.random.normal(shape=(hidden_size, 3 * hidden_size)).astype(mx.float16)
            W_o = mx.random.normal(shape=(hidden_size, hidden_size)).astype(mx.float16)
            W_up = mx.random.normal(shape=(hidden_size, intermediate_size)).astype(mx.float16)
            W_gate = mx.random.normal(shape=(hidden_size, intermediate_size)).astype(mx.float16)
            W_down = mx.random.normal(shape=(intermediate_size, hidden_size)).astype(mx.float16)
            
            x = mx.random.normal(shape=(batch_size, seq_len, hidden_size)).astype(mx.float16)
            mx.eval(W_qkv, W_o, W_up, W_gate, W_down, x)
            
            scale = 1.0 / (head_dim ** 0.5)
            
            def transformer_block(x):
                B, S, H = x.shape
                
                # QKV
                qkv = x @ W_qkv
                q, k, v = mx.split(qkv, 3, axis=-1)
                
                # Reshape
                q = mx.reshape(q, (B, S, num_heads, head_dim))
                q = mx.transpose(q, (0, 2, 1, 3))
                k = mx.reshape(k, (B, S, num_heads, head_dim))
                k = mx.transpose(k, (0, 2, 1, 3))
                v = mx.reshape(v, (B, S, num_heads, head_dim))
                v = mx.transpose(v, (0, 2, 1, 3))
                
                # Attention
                scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale
                probs = mx.softmax(scores, axis=-1)
                attn_out = probs @ v
                
                # Reshape back
                attn_out = mx.transpose(attn_out, (0, 2, 1, 3))
                attn_out = mx.reshape(attn_out, (B, S, H))
                
                # Output projection + residual
                attn_out = attn_out @ W_o
                x = x + attn_out
                
                # FFN (SwiGLU)
                up = x @ W_up
                gate = x @ W_gate
                ffn_out = nn.silu(gate) * up
                ffn_out = ffn_out @ W_down
                x = x + ffn_out
                
                return x
            
            # Warmup
            for _ in range(warmup_iters):
                out = transformer_block(x)
                mx.eval(out)
            
            # Benchmark
            times = []
            for _ in range(bench_iters):
                start = time.perf_counter()
                out = transformer_block(x)
                mx.eval(out)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            
            # FLOPS
            qkv_flops = 2 * batch_size * seq_len * hidden_size * 3 * hidden_size
            attn_flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
            o_flops = 2 * batch_size * seq_len * hidden_size * hidden_size
            ffn_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size * 3
            
            total_flops = qkv_flops + attn_flops + o_flops + ffn_flops
            tflops = (total_flops / avg_time) / 1e9
            
            result = BenchmarkResult(
                name=f"transformer_block_seq{seq_len}",
                size=f"seq={seq_len}, hidden={hidden_size}",
                dtype="float16",
                time_ms=avg_time,
                tflops=tflops,
                extra={
                    "hidden_size": hidden_size,
                    "num_heads": num_heads,
                    "intermediate_size": intermediate_size,
                    "seq_len": seq_len,
                }
            )
            results.append(result)
            
            print(f"  Transformer Block seq={seq_len:4d}: {avg_time:8.2f}ms  {tflops:6.2f} TFLOPS")
            
            clear_memory()
            
        except Exception as e:
            print(f"  Transformer Block seq={seq_len}: FAILED - {e}")
            results.append(BenchmarkResult(
                name=f"transformer_block_seq{seq_len}",
                extra={"error": str(e)}
            ))
    
    return results

# ============================================================================
# MODEL-LEVEL BENCHMARK (using mlx-lm)
# ============================================================================

def benchmark_mistral_mlx(
    model_name: str = "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
    seq_lengths: List[int] = [128, 256, 512],
    warmup_iters: int = 2,
    bench_iters: int = 5,
) -> List[BenchmarkResult]:
    """Benchmark Mistral using MLX-LM (uses quantized models)."""
    
    if not MLX_AVAILABLE:
        return [BenchmarkResult(name="mistral_mlx", extra={"error": "MLX not available"})]
    
    results = []
    
    print("\n" + "="*60)
    print("MISTRAL-7B INFERENCE BENCHMARK (MLX-LM)")
    print("="*60)
    
    try:
        from mlx_lm import load, generate
        
        print(f"Loading {model_name}...")
        print("(Using 4-bit quantized version for MLX)")
        
        load_start = time.time()
        model, tokenizer = load(model_name)
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.1f}s")
        
        for seq_len in seq_lengths:
            try:
                # Create prompt that generates roughly seq_len tokens
                prompt = "The quick brown fox " * (seq_len // 5)
                prompt = prompt[:seq_len * 4]  # Rough token estimate
                
                # Warmup
                for _ in range(warmup_iters):
                    _ = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
                
                # Benchmark generation speed
                max_new_tokens = 50
                times = []
                
                for _ in range(bench_iters):
                    start = time.perf_counter()
                    output = generate(model, tokenizer, prompt=prompt, 
                                     max_tokens=max_new_tokens, verbose=False)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
                
                avg_time = np.mean(times)
                tokens_per_sec = max_new_tokens / (avg_time / 1000)
                
                result = BenchmarkResult(
                    name=f"mistral_mlx_seq{seq_len}",
                    size=f"prompt≈{seq_len}, gen={max_new_tokens}",
                    dtype="4bit",
                    time_ms=avg_time,
                    extra={
                        "tokens_per_sec": tokens_per_sec,
                        "max_new_tokens": max_new_tokens,
                        "model_name": model_name,
                    }
                )
                results.append(result)
                
                print(f"  MLX Mistral prompt≈{seq_len:4d}: {avg_time:8.1f}ms  "
                      f"{tokens_per_sec:6.1f} tok/s (generation)")
                
            except Exception as e:
                print(f"  MLX Mistral prompt≈{seq_len}: FAILED - {e}")
                results.append(BenchmarkResult(
                    name=f"mistral_mlx_seq{seq_len}",
                    extra={"error": str(e)}
                ))
        
        del model
        clear_memory()
        
    except ImportError:
        print("mlx-lm not installed. Install with: pip install mlx-lm")
        results.append(BenchmarkResult(
            name="mistral_mlx",
            extra={"error": "mlx-lm not installed"}
        ))
    except Exception as e:
        print(f"Failed: {e}")
        results.append(BenchmarkResult(
            name="mistral_mlx",
            extra={"error": str(e)}
        ))
    
    return results

# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_mlx_benchmarks(config: Dict = None) -> Dict[str, Any]:
    """Run all MLX benchmarks."""
    
    print("\n" + "="*70)
    print("       MLX BENCHMARK SUITE")
    print("="*70)
    
    if not MLX_AVAILABLE:
        print("ERROR: MLX not available!")
        return {"error": "MLX not installed"}
    
    print(f"Device: {mx.default_device()}")
    
    all_results = {
        "framework": "mlx",
        "device": str(mx.default_device()),
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {}
    }
    
    # GEMM
    gemm_results = benchmark_gemm_mlx()
    all_results["benchmarks"]["gemm"] = [r.to_dict() for r in gemm_results]
    
    # Attention
    attn_results = benchmark_attention_mlx()
    all_results["benchmarks"]["attention"] = [r.to_dict() for r in attn_results]
    
    # Transformer Block
    block_results = benchmark_transformer_block_mlx()
    all_results["benchmarks"]["transformer_block"] = [r.to_dict() for r in block_results]
    
    # Mistral with MLX-LM (optional)
    print("\n[Note: MLX Mistral benchmark uses 4-bit quantized model from mlx-community]")
    user_input = input("Run MLX Mistral benchmark? (y/n): ").strip().lower()
    if user_input == 'y':
        mistral_results = benchmark_mistral_mlx()
        all_results["benchmarks"]["mistral_mlx"] = [r.to_dict() for r in mistral_results]
    
    return all_results


if __name__ == "__main__":
    results = run_all_mlx_benchmarks()
    
    # Save results
    import os
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/mlx_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Results saved to: results/raw/mlx_benchmarks.json")
    print("="*60)
