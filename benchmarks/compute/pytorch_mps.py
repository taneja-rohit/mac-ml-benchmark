"""
PyTorch + MPS (Apple Silicon GPU) Compute Benchmarks

Measures:
- GEMM (matrix multiply) performance at various sizes
- Attention performance
- Transformer block performance
- Model-level benchmarks (Mistral-7B)
"""

import torch
import time
import json
import gc
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    framework: str = "pytorch_mps"
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

def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        print("WARNING: MPS not available, falling back to CPU")
        return torch.device("cpu")

def sync_device(device):
    """Synchronize device to ensure accurate timing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def calculate_gemm_flops(M: int, N: int, K: int) -> int:
    """Calculate FLOPS for matrix multiply: C[M,N] = A[M,K] @ B[K,N]."""
    return 2 * M * N * K  # multiply + add for each output element

def calculate_gemm_bytes(M: int, N: int, K: int, dtype) -> int:
    """Calculate memory traffic for GEMM."""
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    # Read A, B, write C
    return bytes_per_elem * (M * K + K * N + M * N)

# ============================================================================
# GEMM BENCHMARKS
# ============================================================================

def benchmark_gemm(
    sizes: List[int] = [512, 1024, 2048, 4096, 8192],
    dtypes: List[str] = ["float32", "float16"],
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> List[BenchmarkResult]:
    """Benchmark matrix multiplication at various sizes."""
    
    device = get_device()
    results = []
    
    print("\n" + "="*60)
    print("GEMM BENCHMARK (PyTorch + MPS)")
    print("="*60)
    
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        
        for size in sizes:
            M, N, K = size, size, size
            
            try:
                # Allocate matrices
                A = torch.randn(M, K, dtype=dtype, device=device)
                B = torch.randn(K, N, dtype=dtype, device=device)
                
                # Warmup
                for _ in range(warmup_iters):
                    C = A @ B
                    sync_device(device)
                
                # Benchmark
                times = []
                for _ in range(bench_iters):
                    sync_device(device)
                    start = time.perf_counter()
                    C = A @ B
                    sync_device(device)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # ms
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # Calculate metrics
                flops = calculate_gemm_flops(M, N, K)
                tflops = (flops / avg_time) / 1e9  # TFLOPS
                
                bytes_moved = calculate_gemm_bytes(M, N, K, dtype)
                bandwidth = (bytes_moved / avg_time) / 1e6  # GB/s
                
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

def benchmark_attention(
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> List[BenchmarkResult]:
    """Benchmark self-attention at various sequence lengths."""
    
    device = get_device()
    results = []
    
    print("\n" + "="*60)
    print("ATTENTION BENCHMARK (PyTorch + MPS)")
    print("="*60)
    
    for seq_len in seq_lengths:
        try:
            # Create Q, K, V
            hidden_dim = num_heads * head_dim
            
            Q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                           dtype=torch.float16, device=device)
            K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                           dtype=torch.float16, device=device)
            V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                           dtype=torch.float16, device=device)
            
            scale = 1.0 / (head_dim ** 0.5)
            
            def attention_forward():
                # Q @ K.T
                scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                # Softmax
                probs = torch.softmax(scores, dim=-1)
                # @ V
                output = torch.matmul(probs, V)
                return output
            
            # Warmup
            for _ in range(warmup_iters):
                _ = attention_forward()
                sync_device(device)
            
            # Benchmark
            times = []
            for _ in range(bench_iters):
                sync_device(device)
                start = time.perf_counter()
                _ = attention_forward()
                sync_device(device)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            
            # Calculate FLOPS for attention
            # Q @ K.T: 2 * B * H * S * S * D
            # Softmax: ~5 * B * H * S * S (approx)
            # P @ V: 2 * B * H * S * D * S
            qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
            pv_flops = 2 * batch_size * num_heads * seq_len * head_dim * seq_len
            total_flops = qk_flops + pv_flops
            tflops = (total_flops / avg_time) / 1e9
            
            # Memory for attention scores: B * H * S * S * 2 bytes (fp16)
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

def benchmark_transformer_block(
    hidden_size: int = 4096,
    num_heads: int = 32,
    intermediate_size: int = 14336,  # Mistral's FFN size
    seq_lengths: List[int] = [256, 512, 1024],
    batch_size: int = 1,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> List[BenchmarkResult]:
    """Benchmark a full transformer block (attention + FFN)."""
    
    device = get_device()
    results = []
    
    print("\n" + "="*60)
    print("TRANSFORMER BLOCK BENCHMARK (PyTorch + MPS)")
    print("="*60)
    
    for seq_len in seq_lengths:
        try:
            # Simplified transformer block (no RoPE, no KV cache for simplicity)
            head_dim = hidden_size // num_heads
            
            # Weights
            W_qkv = torch.randn(hidden_size, 3 * hidden_size, dtype=torch.float16, device=device)
            W_o = torch.randn(hidden_size, hidden_size, dtype=torch.float16, device=device)
            W_up = torch.randn(hidden_size, intermediate_size, dtype=torch.float16, device=device)
            W_gate = torch.randn(hidden_size, intermediate_size, dtype=torch.float16, device=device)
            W_down = torch.randn(intermediate_size, hidden_size, dtype=torch.float16, device=device)
            
            # Input
            x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)
            
            scale = 1.0 / (head_dim ** 0.5)
            
            def transformer_block(x):
                B, S, H = x.shape
                
                # QKV projection
                qkv = x @ W_qkv
                q, k, v = qkv.chunk(3, dim=-1)
                
                # Reshape for multi-head attention
                q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
                k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
                v = v.view(B, S, num_heads, head_dim).transpose(1, 2)
                
                # Attention
                scores = (q @ k.transpose(-2, -1)) * scale
                probs = torch.softmax(scores, dim=-1)
                attn_out = probs @ v
                
                # Reshape back
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H)
                
                # Output projection
                attn_out = attn_out @ W_o
                
                # Residual (simplified)
                x = x + attn_out
                
                # FFN (SwiGLU-style)
                up = x @ W_up
                gate = x @ W_gate
                ffn_out = torch.nn.functional.silu(gate) * up
                ffn_out = ffn_out @ W_down
                
                # Residual
                x = x + ffn_out
                
                return x
            
            # Warmup
            for _ in range(warmup_iters):
                _ = transformer_block(x)
                sync_device(device)
            
            # Benchmark
            times = []
            for _ in range(bench_iters):
                sync_device(device)
                start = time.perf_counter()
                _ = transformer_block(x)
                sync_device(device)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            
            # Rough FLOPS calculation for transformer block
            # QKV: 2 * B * S * H * 3H
            # Attention: 2 * B * heads * S * S * head_dim * 2
            # O proj: 2 * B * S * H * H
            # FFN up/gate: 2 * B * S * H * I * 2
            # FFN down: 2 * B * S * I * H
            
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
# MODEL-LEVEL BENCHMARK (Mistral-7B)
# ============================================================================

def benchmark_mistral_inference(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    seq_lengths: List[int] = [128, 256, 512],
    batch_size: int = 1,
    warmup_iters: int = 2,
    bench_iters: int = 5,
) -> List[BenchmarkResult]:
    """Benchmark Mistral-7B inference."""
    
    device = get_device()
    results = []
    
    print("\n" + "="*60)
    print("MISTRAL-7B INFERENCE BENCHMARK (PyTorch + MPS)")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {model_name}...")
        load_start = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="mps" if device.type == "mps" else "auto",
            low_cpu_mem_usage=True,
        )
        
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.1f}s")
        
        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        param_gb = (param_count * 2) / (1024**3)  # fp16
        print(f"Parameters: {param_count/1e9:.2f}B ({param_gb:.1f} GB in fp16)")
        
        model.eval()
        
        for seq_len in seq_lengths:
            try:
                # Create dummy input
                input_ids = torch.randint(0, tokenizer.vocab_size, 
                                         (batch_size, seq_len), device=device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(warmup_iters):
                        _ = model(input_ids)
                        sync_device(device)
                
                # Benchmark forward pass
                times = []
                with torch.no_grad():
                    for _ in range(bench_iters):
                        sync_device(device)
                        start = time.perf_counter()
                        outputs = model(input_ids)
                        sync_device(device)
                        end = time.perf_counter()
                        times.append((end - start) * 1000)
                
                avg_time = np.mean(times)
                
                # Tokens per second
                tokens_per_sec = (batch_size * seq_len) / (avg_time / 1000)
                
                # Rough FLOPS estimate for 7B model
                # ~2 * params * tokens for forward pass
                flops_per_token = 2 * param_count
                total_flops = flops_per_token * batch_size * seq_len
                tflops = (total_flops / avg_time) / 1e9
                
                # MFU calculation (assumes ~15 TFLOPS peak for M5 Pro)
                # This will be calibrated after GEMM benchmarks
                peak_tflops = 15  # Estimate, will refine
                mfu = (tflops / peak_tflops) * 100
                
                result = BenchmarkResult(
                    name=f"mistral_7b_seq{seq_len}",
                    size=f"seq={seq_len}, params=7B",
                    dtype="float16",
                    time_ms=avg_time,
                    tflops=tflops,
                    utilization_pct=mfu,
                    extra={
                        "tokens_per_sec": tokens_per_sec,
                        "param_count": param_count,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "model_name": model_name,
                    }
                )
                results.append(result)
                
                print(f"  Mistral-7B seq={seq_len:4d}: {avg_time:8.1f}ms  "
                      f"{tokens_per_sec:6.0f} tok/s  {tflops:.2f} TFLOPS  MFU≈{mfu:.1f}%")
                
            except Exception as e:
                print(f"  Mistral-7B seq={seq_len}: FAILED - {e}")
                results.append(BenchmarkResult(
                    name=f"mistral_7b_seq{seq_len}",
                    extra={"error": str(e)}
                ))
        
        # Clean up
        del model
        clear_memory()
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        results.append(BenchmarkResult(
            name="mistral_7b",
            extra={"error": str(e)}
        ))
    
    return results

# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_pytorch_benchmarks(config: Dict = None) -> Dict[str, List[BenchmarkResult]]:
    """Run all PyTorch + MPS benchmarks."""
    
    print("\n" + "="*70)
    print("       PYTORCH + MPS BENCHMARK SUITE")
    print("="*70)
    
    device = get_device()
    print(f"Device: {device}")
    
    all_results = {
        "framework": "pytorch_mps",
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {}
    }
    
    # GEMM
    gemm_results = benchmark_gemm()
    all_results["benchmarks"]["gemm"] = [r.to_dict() for r in gemm_results]
    
    # Attention
    attn_results = benchmark_attention()
    all_results["benchmarks"]["attention"] = [r.to_dict() for r in attn_results]
    
    # Transformer Block
    block_results = benchmark_transformer_block()
    all_results["benchmarks"]["transformer_block"] = [r.to_dict() for r in block_results]
    
    # Mistral-7B (optional, takes longer)
    print("\n[Note: Mistral-7B benchmark will download ~14GB model on first run]")
    user_input = input("Run Mistral-7B benchmark? (y/n): ").strip().lower()
    if user_input == 'y':
        mistral_results = benchmark_mistral_inference()
        all_results["benchmarks"]["mistral_7b"] = [r.to_dict() for r in mistral_results]
    
    return all_results


if __name__ == "__main__":
    results = run_all_pytorch_benchmarks()
    
    # Save results
    import os
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/pytorch_mps_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Results saved to: results/raw/pytorch_mps_benchmarks.json")
    print("="*60)
