"""
Memory Bandwidth Benchmarks for Apple Silicon

Measures actual memory bandwidth vs theoretical.
Apple Silicon unified memory: ~200-400 GB/s depending on chip.
"""

import torch
import time
import json
import gc
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

@dataclass
class BenchmarkResult:
    name: str
    framework: str = "pytorch_mps"
    size_gb: float = 0.0
    time_ms: float = 0.0
    bandwidth_gbs: float = 0.0
    extra: Dict[str, Any] = None
    
    def to_dict(self):
        return asdict(self)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sync_device(device):
    if device.type == "mps":
        torch.mps.synchronize()

def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def benchmark_memory_bandwidth(
    sizes_gb: List[float] = [0.1, 0.25, 0.5, 1.0, 1.5],  # Smaller sizes for MPS limits
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> List[BenchmarkResult]:
    """Measure memory bandwidth with various operations."""
    
    device = get_device()
    results = []
    
    print("\n" + "="*60)
    print("MEMORY BANDWIDTH BENCHMARK")
    print("="*60)
    print(f"Device: {device}")
    
    for size_gb in sizes_gb:
        size_bytes = int(size_gb * 1024**3)
        num_elements = size_bytes // 4  # float32
        
        # Use 2D shape to avoid INT_MAX limit on MPS
        side = int(num_elements ** 0.5)
        actual_elements = side * side
        actual_size_gb = (actual_elements * 4) / (1024**3)
        
        try:
            # Sequential Read (sum reduction)
            print(f"\n  Size: {actual_size_gb:.2f} GB ({side}x{side})")
            
            A = torch.randn(side, side, dtype=torch.float32, device=device)
            sync_device(device)
            
            # Warmup
            for _ in range(warmup_iters):
                _ = A.sum()
                sync_device(device)
            
            # Benchmark read
            times = []
            for _ in range(bench_iters):
                sync_device(device)
                start = time.perf_counter()
                _ = A.sum()
                sync_device(device)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            actual_bytes = actual_elements * 4
            read_bw = (actual_bytes / avg_time) / 1e6  # GB/s
            
            results.append(BenchmarkResult(
                name="sequential_read",
                size_gb=actual_size_gb,
                time_ms=avg_time,
                bandwidth_gbs=read_bw,
            ))
            print(f"    Read:  {avg_time:6.2f}ms  {read_bw:6.1f} GB/s")
            
            # Sequential Write (fill)
            for _ in range(warmup_iters):
                A.fill_(1.0)
                sync_device(device)
            
            times = []
            for _ in range(bench_iters):
                sync_device(device)
                start = time.perf_counter()
                A.fill_(1.0)
                sync_device(device)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            write_bw = (actual_bytes / avg_time) / 1e6
            
            results.append(BenchmarkResult(
                name="sequential_write",
                size_gb=actual_size_gb,
                time_ms=avg_time,
                bandwidth_gbs=write_bw,
            ))
            print(f"    Write: {avg_time:6.2f}ms  {write_bw:6.1f} GB/s")
            
            # Copy (Read + Write)
            B = torch.empty_like(A)
            sync_device(device)
            
            for _ in range(warmup_iters):
                B.copy_(A)
                sync_device(device)
            
            times = []
            for _ in range(bench_iters):
                sync_device(device)
                start = time.perf_counter()
                B.copy_(A)
                sync_device(device)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            copy_bw = (2 * actual_bytes / avg_time) / 1e6  # Read + Write
            
            results.append(BenchmarkResult(
                name="copy",
                size_gb=actual_size_gb,
                time_ms=avg_time,
                bandwidth_gbs=copy_bw,
            ))
            print(f"    Copy:  {avg_time:6.2f}ms  {copy_bw:6.1f} GB/s (R+W)")
            
            del A, B
            clear_memory()
            
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append(BenchmarkResult(
                name="memory_bandwidth",
                size_gb=size_gb,
                extra={"error": str(e)}
            ))
    
    return results

def benchmark_max_allocation() -> BenchmarkResult:
    """Find maximum tensor allocation size using 2D tensors."""
    
    device = get_device()
    print("\n" + "="*60)
    print("MAX ALLOCATION TEST")
    print("="*60)
    
    max_size_gb = 0
    
    # Test incrementally with 2D tensors to avoid INT_MAX limit
    for size_gb in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]:
        try:
            size_bytes = int(size_gb * 1024**3)
            num_elements = size_bytes // 4
            side = int(num_elements ** 0.5)
            
            A = torch.randn(side, side, dtype=torch.float32, device=device)
            sync_device(device)
            
            actual_gb = (side * side * 4) / (1024**3)
            max_size_gb = actual_gb
            print(f"  ✓ {actual_gb:.1f} GB ({side}x{side}) - Success")
            
            del A
            clear_memory()
                
        except Exception as e:
            print(f"  ✗ {size_gb:.1f} GB - Failed ({type(e).__name__})")
            break
    
    print(f"\n  Maximum allocation: {max_size_gb:.1f} GB")
    
    return BenchmarkResult(
        name="max_allocation",
        size_gb=max_size_gb,
        extra={"device": str(device)}
    )

def run_all_memory_benchmarks() -> Dict[str, Any]:
    """Run all memory benchmarks."""
    
    print("\n" + "="*70)
    print("       MEMORY BENCHMARK SUITE")
    print("="*70)
    
    all_results = {
        "framework": "pytorch_mps",
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {}
    }
    
    # Bandwidth
    bw_results = benchmark_memory_bandwidth()
    all_results["benchmarks"]["bandwidth"] = [r.to_dict() for r in bw_results]
    
    # Max allocation
    max_result = benchmark_max_allocation()
    all_results["benchmarks"]["max_allocation"] = max_result.to_dict()
    
    return all_results

if __name__ == "__main__":
    results = run_all_memory_benchmarks()
    
    import os
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/memory_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: results/raw/memory_benchmarks.json")
