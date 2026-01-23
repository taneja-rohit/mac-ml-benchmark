#!/usr/bin/env python3
"""
Mac ML Benchmark Suite - Main Runner

This is the glue script that orchestrates all benchmarks.

Usage:
    python run_benchmarks.py                    # Run all benchmarks (includes fine-tuning)
    python run_benchmarks.py --discovery-only   # Just hardware discovery
    python run_benchmarks.py --pytorch-only     # Just PyTorch benchmarks
    python run_benchmarks.py --mlx-only         # Just MLX benchmarks
    python run_benchmarks.py --memory-only      # Just memory benchmarks
    python run_benchmarks.py --finetuning-only  # Just fine-tuning benchmarks
    python run_benchmarks.py --layer-only       # Just layer benchmarks (float16 fwd+bwd)
    python run_benchmarks.py --skip-finetuning  # Run all except fine-tuning
    python run_benchmarks.py --quick            # Quick mode (smaller sizes)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     🍎  MAC ML BENCHMARK SUITE  🍎                                   ║
║                                                                      ║
║     Comprehensive ML benchmarks for Apple Silicon                    ║
║     PyTorch + MPS  |  MLX  |  Memory  |  Compute                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def run_discovery():
    """Run hardware discovery."""
    print("\n" + "="*70)
    print("PHASE 0: HARDWARE DISCOVERY")
    print("="*70)
    
    from discovery.system_info import get_system_info, print_system_info, save_system_info
    
    info = get_system_info()
    print_system_info(info)
    
    # Save
    os.makedirs("results/raw", exist_ok=True)
    save_system_info(info, "results/raw/system_info.json")
    
    return info

def run_pytorch_benchmarks(quick: bool = False):
    """Run PyTorch + MPS benchmarks."""
    print("\n" + "="*70)
    print("PHASE 1: PYTORCH + MPS BENCHMARKS")
    print("="*70)
    
    from benchmarks.compute.pytorch_mps import run_all_pytorch_benchmarks
    
    results = run_all_pytorch_benchmarks()
    
    # Save
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/pytorch_mps_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nPyTorch results saved to: results/raw/pytorch_mps_benchmarks.json")
    return results

def run_mlx_benchmarks(quick: bool = False):
    """Run MLX benchmarks."""
    print("\n" + "="*70)
    print("PHASE 2: MLX BENCHMARKS")
    print("="*70)
    
    try:
        from benchmarks.compute.mlx_benchmarks import run_all_mlx_benchmarks
        
        results = run_all_mlx_benchmarks()
        
        # Save
        os.makedirs("results/raw", exist_ok=True)
        with open("results/raw/mlx_benchmarks.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\nMLX results saved to: results/raw/mlx_benchmarks.json")
        return results
        
    except ImportError as e:
        print(f"MLX not available: {e}")
        print("Install with: pip install mlx mlx-lm")
        return {"error": str(e)}

def run_memory_benchmarks():
    """Run memory benchmarks."""
    print("\n" + "="*70)
    print("PHASE 3: MEMORY BENCHMARKS")
    print("="*70)
    
    from benchmarks.memory.bandwidth import run_all_memory_benchmarks
    
    results = run_all_memory_benchmarks()
    
    # Save
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/memory_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nMemory results saved to: results/raw/memory_benchmarks.json")
    return results

def run_finetuning_benchmarks(raw_dir: str):
    """Run fine-tuning benchmarks for both PyTorch and MLX."""
    print("\n" + "="*70)
    print("PHASE 4: FINE-TUNING BENCHMARKS (float16)")
    print("="*70)
    
    results = {}
    
    # PyTorch + MPS fine-tuning
    print("\n[4a] PyTorch + MPS Fine-tuning (float16)")
    user_input = input("Run PyTorch fine-tuning benchmark? (y/n): ").strip().lower()
    if user_input == 'y':
        from benchmarks.compute.finetuning import benchmark_pytorch_finetuning
        pytorch_ft = benchmark_pytorch_finetuning()
        results["pytorch_mps"] = pytorch_ft
        
        with open(f"{raw_dir}/pytorch_mps_finetuning.json", "w") as f:
            json.dump(pytorch_ft, f, indent=2)
        print(f"Saved to: {raw_dir}/pytorch_mps_finetuning.json")
    
    # MLX LoRA fine-tuning
    print("\n[4b] MLX LoRA Fine-tuning (4-bit)")
    user_input = input("Run MLX LoRA fine-tuning benchmark? (y/n): ").strip().lower()
    if user_input == 'y':
        import subprocess
        print("Running mlx_lm.lora...")
        adapter_path = f"{raw_dir}/lora_adapters"
        cmd = [
            "mlx_lm.lora",
            "--model", "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
            "--train",
            "--data", "./data",
            "--iters", "50",
            "--adapter-path", adapter_path,
            "--batch-size", "1"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            # Parse output for tokens/sec
            results["mlx_lora"] = {
                "model": "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
                "precision": "4-bit quantized",
                "adapter_path": adapter_path,
                "status": "completed"
            }
            print(f"MLX LoRA adapters saved to: {adapter_path}")
        else:
            print(f"MLX LoRA failed: {result.stderr}")
    
    return results

def run_layer_benchmarks(raw_dir: str):
    """Run layer-level float16 benchmarks comparing PyTorch and MLX."""
    print("\n" + "="*70)
    print("PHASE 5: LAYER BENCHMARKS (float16 forward+backward)")
    print("="*70)
    
    user_input = input("Run layer benchmark (compares training performance)? (y/n): ").strip().lower()
    if user_input != 'y':
        return None
    
    from benchmarks.compute.layer_benchmark import run_layer_benchmark
    results = run_layer_benchmark(f"{raw_dir}/layer_benchmark_float16.json")
    return results

def generate_summary_report(system_info, pytorch_results, mlx_results, memory_results):
    """Generate a summary report."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("MAC ML BENCHMARK SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("="*70)
    
    # Hardware
    report.append("\n## HARDWARE")
    chip = system_info.get('chip', {})
    mem = system_info.get('memory', {})
    gpu = system_info.get('gpu', {})
    
    report.append(f"  Chip: {chip.get('chip_name', 'Unknown')}")
    report.append(f"  CPU Cores: {chip.get('core_count', 'Unknown')}")
    report.append(f"  GPU Cores: {gpu.get('gpu_cores_estimate', 'Unknown')}")
    report.append(f"  Memory: {mem.get('total_gb', 0):.0f} GB")
    report.append(f"  MPS Available: {gpu.get('mps_available', False)}")
    report.append(f"  MLX Available: {gpu.get('mlx_available', False)}")
    
    # PyTorch Results
    if pytorch_results and 'benchmarks' in pytorch_results:
        report.append("\n## PYTORCH + MPS RESULTS")
        
        # GEMM
        if 'gemm' in pytorch_results['benchmarks']:
            report.append("\n  GEMM Performance:")
            for r in pytorch_results['benchmarks']['gemm']:
                if 'error' not in (r.get('extra') or {}):
                    report.append(f"    {r['size']:12s} ({r['dtype']:8s}): "
                                 f"{r['tflops']:.2f} TFLOPS")
        
        # Find peak TFLOPS
        all_tflops = []
        for bench in pytorch_results['benchmarks'].values():
            if isinstance(bench, list):
                for r in bench:
                    if r.get('tflops', 0) > 0:
                        all_tflops.append(r['tflops'])
        
        if all_tflops:
            peak = max(all_tflops)
            report.append(f"\n  Peak Achieved: {peak:.2f} TFLOPS (PyTorch+MPS)")
    
    # MLX Results
    if mlx_results and 'benchmarks' in mlx_results:
        report.append("\n## MLX RESULTS")
        
        if 'gemm' in mlx_results['benchmarks']:
            report.append("\n  GEMM Performance:")
            for r in mlx_results['benchmarks']['gemm']:
                if 'error' not in (r.get('extra') or {}):
                    report.append(f"    {r['size']:12s} ({r['dtype']:8s}): "
                                 f"{r['tflops']:.2f} TFLOPS")
        
        all_tflops = []
        for bench in mlx_results['benchmarks'].values():
            if isinstance(bench, list):
                for r in bench:
                    if r.get('tflops', 0) > 0:
                        all_tflops.append(r['tflops'])
        
        if all_tflops:
            peak = max(all_tflops)
            report.append(f"\n  Peak Achieved: {peak:.2f} TFLOPS (MLX)")
    
    # Memory Results
    if memory_results and 'benchmarks' in memory_results:
        report.append("\n## MEMORY BANDWIDTH")
        
        if 'bandwidth' in memory_results['benchmarks']:
            # Find peak bandwidth
            read_bw = [r['bandwidth_gbs'] for r in memory_results['benchmarks']['bandwidth']
                      if r.get('name') == 'sequential_read' and r.get('bandwidth_gbs', 0) > 0]
            copy_bw = [r['bandwidth_gbs'] for r in memory_results['benchmarks']['bandwidth']
                      if r.get('name') == 'copy' and r.get('bandwidth_gbs', 0) > 0]
            
            if read_bw:
                report.append(f"  Peak Read Bandwidth: {max(read_bw):.1f} GB/s")
            if copy_bw:
                report.append(f"  Peak Copy Bandwidth: {max(copy_bw):.1f} GB/s")
        
        if 'max_allocation' in memory_results['benchmarks']:
            max_alloc = memory_results['benchmarks']['max_allocation']
            report.append(f"  Max GPU Allocation: {max_alloc.get('size_gb', 0):.1f} GB")
    
    # Framework Comparison
    report.append("\n## FRAMEWORK COMPARISON")
    report.append("  (Compare GEMM TFLOPS at same sizes between PyTorch and MLX)")
    
    report.append("\n" + "="*70)
    
    # Print and save
    report_text = "\n".join(report)
    print(report_text)
    
    os.makedirs("results/reports", exist_ok=True)
    with open("results/reports/summary.txt", "w") as f:
        f.write(report_text)
    
    print("\nReport saved to: results/reports/summary.txt")

def get_machine_folder(system_info):
    """Determine the folder name based on the chip."""
    chip = system_info.get('chip', {}).get('chip_name', 'Unknown')
    if 'M5' in chip:
        return 'M5'
    if 'M4 Pro' in chip:
        return 'M4_Pro'
    return chip.replace(' ', '_')

def main():
    parser = argparse.ArgumentParser(description="Mac ML Benchmark Suite")
    parser.add_argument("--discovery-only", action="store_true", 
                        help="Just run hardware discovery")
    parser.add_argument("--pytorch-only", action="store_true",
                        help="Just run PyTorch benchmarks")
    parser.add_argument("--mlx-only", action="store_true",
                        help="Just run MLX benchmarks")
    parser.add_argument("--memory-only", action="store_true",
                        help="Just run memory benchmarks")
    parser.add_argument("--finetuning-only", action="store_true",
                        help="Just run fine-tuning benchmarks")
    parser.add_argument("--layer-only", action="store_true",
                        help="Just run layer benchmarks")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (smaller sizes)")
    parser.add_argument("--skip-finetuning", action="store_true",
                        help="Skip fine-tuning benchmarks in full run")
    args = parser.parse_args()
    
    print_banner()
    
    # Always run discovery
    system_info = run_discovery()
    machine_folder = get_machine_folder(system_info)
    raw_dir = f"results/raw/{machine_folder}"
    report_dir = f"results/reports/{machine_folder}"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # Save system info to machine folder
    with open(f"{raw_dir}/system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)
    
    if args.discovery_only:
        print(f"\n✅ Hardware discovery complete! Saved to {raw_dir}")
        return
    
    pytorch_results = None
    mlx_results = None
    memory_results = None
    
    # Run selected benchmarks
    if args.pytorch_only:
        pytorch_results = run_pytorch_benchmarks(quick=args.quick)
    elif args.mlx_only:
        mlx_results = run_mlx_benchmarks(quick=args.quick)
    elif args.memory_only:
        memory_results = run_memory_benchmarks()
    elif args.finetuning_only:
        run_finetuning_benchmarks(raw_dir)
    elif args.layer_only:
        run_layer_benchmarks(raw_dir)
    else:
        # Run all compute benchmarks
        pytorch_results = run_pytorch_benchmarks(quick=args.quick)
        mlx_results = run_mlx_benchmarks(quick=args.quick)
        memory_results = run_memory_benchmarks()
        
        # Run fine-tuning and layer benchmarks
        if not args.skip_finetuning:
            run_finetuning_benchmarks(raw_dir)
            run_layer_benchmarks(raw_dir)
    
    # Generate summary
    generate_summary_report(system_info, pytorch_results, mlx_results, memory_results)
    
    print("\n" + "="*70)
    print("✅ BENCHMARKS COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {raw_dir}/")
    print("  - system_info.json")
    print("  - pytorch_mps_benchmarks.json")
    print("  - mlx_benchmarks.json")
    print("  - memory_benchmarks.json")
    print("  - pytorch_mps_finetuning.json")
    print("  - layer_benchmark_float16.json")
    print("  - lora_adapters/")

if __name__ == "__main__":
    main()
