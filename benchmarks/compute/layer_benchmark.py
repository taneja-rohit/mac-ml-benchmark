#!/usr/bin/env python3
"""
Layer-level float16 benchmark comparing PyTorch+MPS vs MLX.
Tests forward and backward pass for attention and FFN layers.
"""

import json
import time
from typing import Dict, Any

def benchmark_pytorch_layers() -> Dict[str, Any]:
    """Benchmark PyTorch transformer layers."""
    import torch
    import torch.nn as nn
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"PyTorch device: {device}")
    
    results = {}
    
    # Mistral-7B config
    hidden_size = 4096
    num_heads = 32
    head_dim = hidden_size // num_heads
    intermediate_size = 14336
    
    for seq_len in [256, 512, 1024]:
        print(f"\n  Seq length: {seq_len}")
        batch_size = 1
        
        # Create input
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device, requires_grad=True)
        
        # Attention benchmark
        q_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(device).half()
        k_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(device).half()
        v_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(device).half()
        o_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(device).half()
        
        # Warmup
        for _ in range(3):
            q = q_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            out = o_proj(out)
            torch.mps.synchronize()
        
        # Forward timing
        times = []
        for _ in range(10):
            torch.mps.synchronize()
            start = time.perf_counter()
            q = q_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            out = o_proj(out)
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        attn_fwd_ms = sum(times) / len(times)
        
        # Backward timing
        times = []
        for _ in range(10):
            x_fresh = x.detach().clone().requires_grad_(True)
            q = q_proj(x_fresh).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k_proj(x_fresh).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v_proj(x_fresh).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            out = o_proj(out)
            torch.mps.synchronize()
            
            start = time.perf_counter()
            out.sum().backward()
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        attn_bwd_ms = sum(times) / len(times)
        
        # FFN benchmark
        gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(device).half()
        up_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(device).half()
        down_proj = nn.Linear(intermediate_size, hidden_size, bias=False).to(device).half()
        
        # Warmup
        for _ in range(3):
            h = torch.nn.functional.silu(gate_proj(x)) * up_proj(x)
            out = down_proj(h)
            torch.mps.synchronize()
        
        # FFN Forward
        times = []
        for _ in range(10):
            torch.mps.synchronize()
            start = time.perf_counter()
            h = torch.nn.functional.silu(gate_proj(x)) * up_proj(x)
            out = down_proj(h)
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        ffn_fwd_ms = sum(times) / len(times)
        
        # FFN Backward
        times = []
        for _ in range(10):
            x_fresh = x.detach().clone().requires_grad_(True)
            h = torch.nn.functional.silu(gate_proj(x_fresh)) * up_proj(x_fresh)
            out = down_proj(h)
            torch.mps.synchronize()
            
            start = time.perf_counter()
            out.sum().backward()
            torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        ffn_bwd_ms = sum(times) / len(times)
        
        # FLOPS calculation
        # Attention: 4 * hidden^2 * seq (projections) + 2 * seq^2 * hidden (QK, AV matmuls)
        attn_flops = 4 * hidden_size * hidden_size * seq_len + 2 * seq_len * seq_len * hidden_size
        attn_tflops = (attn_flops / (attn_fwd_ms / 1000)) / 1e12
        
        # FFN: 3 * hidden * intermediate * seq
        ffn_flops = 3 * hidden_size * intermediate_size * seq_len * 2  # *2 for mul ops
        ffn_tflops = (ffn_flops / (ffn_fwd_ms / 1000)) / 1e12
        
        results[str(seq_len)] = {
            "attention_ms": round(attn_fwd_ms, 2),
            "attention_tflops": round(attn_tflops, 2),
            "attention_backward_ms": round(attn_bwd_ms, 2),
            "ffn_ms": round(ffn_fwd_ms, 2),
            "ffn_tflops": round(ffn_tflops, 2),
            "ffn_backward_ms": round(ffn_bwd_ms, 2),
            "full_layer_ms": round(attn_fwd_ms + ffn_fwd_ms, 2),
            "backward_ms": round(attn_bwd_ms + ffn_bwd_ms, 2),
        }
        
        print(f"    Attention: {attn_fwd_ms:.2f}ms fwd, {attn_bwd_ms:.2f}ms bwd ({attn_tflops:.2f} TFLOPS)")
        print(f"    FFN: {ffn_fwd_ms:.2f}ms fwd, {ffn_bwd_ms:.2f}ms bwd ({ffn_tflops:.2f} TFLOPS)")
    
    return results


def benchmark_mlx_layers() -> Dict[str, Any]:
    """Benchmark MLX transformer layers."""
    import mlx.core as mx
    import mlx.nn as nn
    
    print(f"MLX device: {mx.default_device()}")
    
    results = {}
    
    # Mistral-7B config
    hidden_size = 4096
    num_heads = 32
    head_dim = hidden_size // num_heads
    intermediate_size = 14336
    
    for seq_len in [256, 512, 1024]:
        print(f"\n  Seq length: {seq_len}")
        batch_size = 1
        
        # Create input
        x = mx.random.normal((batch_size, seq_len, hidden_size)).astype(mx.float16)
        
        # Attention layers
        q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        def attention_forward(x):
            B, L, _ = x.shape
            q = q_proj(x).reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k_proj(x).reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v_proj(x).reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / (head_dim ** 0.5)
            attn = mx.softmax(scores, axis=-1)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, hidden_size)
            return o_proj(out)
        
        # Warmup
        for _ in range(3):
            out = attention_forward(x)
            mx.eval(out)
        
        # Forward timing
        times = []
        for _ in range(10):
            start = time.perf_counter()
            out = attention_forward(x)
            mx.eval(out)
            times.append((time.perf_counter() - start) * 1000)
        
        attn_fwd_ms = sum(times) / len(times)
        
        # Backward timing (using value_and_grad)
        def loss_fn(x):
            return attention_forward(x).sum()
        
        grad_fn = mx.grad(loss_fn)
        
        # Warmup
        for _ in range(3):
            g = grad_fn(x)
            mx.eval(g)
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            g = grad_fn(x)
            mx.eval(g)
            times.append((time.perf_counter() - start) * 1000)
        
        attn_bwd_ms = sum(times) / len(times)
        
        # FFN layers
        gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        def ffn_forward(x):
            return down_proj(nn.silu(gate_proj(x)) * up_proj(x))
        
        # Warmup
        for _ in range(3):
            out = ffn_forward(x)
            mx.eval(out)
        
        # Forward timing
        times = []
        for _ in range(10):
            start = time.perf_counter()
            out = ffn_forward(x)
            mx.eval(out)
            times.append((time.perf_counter() - start) * 1000)
        
        ffn_fwd_ms = sum(times) / len(times)
        
        # Backward timing
        def ffn_loss_fn(x):
            return ffn_forward(x).sum()
        
        ffn_grad_fn = mx.grad(ffn_loss_fn)
        
        # Warmup
        for _ in range(3):
            g = ffn_grad_fn(x)
            mx.eval(g)
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            g = ffn_grad_fn(x)
            mx.eval(g)
            times.append((time.perf_counter() - start) * 1000)
        
        ffn_bwd_ms = sum(times) / len(times)
        
        # FLOPS calculation
        attn_flops = 4 * hidden_size * hidden_size * seq_len + 2 * seq_len * seq_len * hidden_size
        attn_tflops = (attn_flops / (attn_fwd_ms / 1000)) / 1e12
        
        ffn_flops = 3 * hidden_size * intermediate_size * seq_len * 2
        ffn_tflops = (ffn_flops / (ffn_fwd_ms / 1000)) / 1e12
        
        results[str(seq_len)] = {
            "attention_ms": round(attn_fwd_ms, 2),
            "attention_tflops": round(attn_tflops, 2),
            "attention_backward_ms": round(attn_bwd_ms, 2),
            "ffn_ms": round(ffn_fwd_ms, 2),
            "ffn_tflops": round(ffn_tflops, 2),
            "ffn_backward_ms": round(ffn_bwd_ms, 2),
            "full_layer_ms": round(attn_fwd_ms + ffn_fwd_ms, 2),
            "backward_ms": round(attn_bwd_ms + ffn_bwd_ms, 2),
        }
        
        print(f"    Attention: {attn_fwd_ms:.2f}ms fwd, {attn_bwd_ms:.2f}ms bwd ({attn_tflops:.2f} TFLOPS)")
        print(f"    FFN: {ffn_fwd_ms:.2f}ms fwd, {ffn_bwd_ms:.2f}ms bwd ({ffn_tflops:.2f} TFLOPS)")
    
    return results


def run_layer_benchmark(output_path: str = None):
    """Run complete layer benchmark."""
    print("="*60)
    print("LAYER BENCHMARK (float16)")
    print("="*60)
    
    print("\n[1/2] PyTorch + MPS")
    pytorch_results = benchmark_pytorch_layers()
    
    print("\n[2/2] MLX")
    mlx_results = benchmark_mlx_layers()
    
    results = {
        "pytorch_mps": pytorch_results,
        "mlx": mlx_results
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/raw/layer_benchmark_float16.json")
    args = parser.parse_args()
    run_layer_benchmark(args.output)
