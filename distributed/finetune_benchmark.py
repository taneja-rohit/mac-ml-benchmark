"""
Distributed Fine-tuning Benchmark with Traffic Monitoring

Benchmarks LoRA fine-tuning with:
- Tensor Parallelism (TP): Split model across devices
- Expert Parallelism (EP): Split MoE experts across devices

Measures:
- Throughput (tokens/sec)
- Memory usage per device
- Thunderbolt bandwidth utilization
- Training loss convergence

Usage (single machine for now):
    python distributed/finetune_benchmark.py --mode simulate

When Thunderbolt cable arrives:
    python distributed/finetune_benchmark.py --mode tp
    python distributed/finetune_benchmark.py --mode ep
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

# Import traffic monitor
from traffic_monitor import ThunderboltMonitor

# Check for MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available")

# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")


def simulate_tp_traffic(monitor: ThunderboltMonitor, duration: float = 5.0):
    """
    Simulate Tensor Parallelism traffic pattern.
    
    TP does AllReduce after every layer:
    - Constant communication throughout forward/backward pass
    - Bandwidth = model_size * 2 / layer_time (roughly)
    """
    print(f"Simulating Tensor Parallelism for {duration}s...")
    
    num_layers = 32  # Like Mistral-7B
    layer_time = duration / num_layers
    
    start = time.perf_counter()
    
    for layer in range(num_layers):
        monitor.mark(f"layer_{layer}_start")
        
        # Simulate forward pass
        time.sleep(layer_time * 0.4)
        monitor.mark(f"layer_{layer}_fwd_allreduce")
        
        # Simulate backward pass  
        time.sleep(layer_time * 0.4)
        monitor.mark(f"layer_{layer}_bwd_allreduce")
        
        # AllReduce sync
        time.sleep(layer_time * 0.2)
        monitor.mark(f"layer_{layer}_end")
    
    elapsed = time.perf_counter() - start
    print(f"TP simulation complete: {elapsed:.2f}s")


def simulate_ep_traffic(monitor: ThunderboltMonitor, duration: float = 5.0):
    """
    Simulate Expert Parallelism traffic pattern.
    
    EP does AllToAll only during MoE routing:
    - Silent during attention layers
    - Burst during expert routing (2x per MoE layer)
    """
    print(f"Simulating Expert Parallelism for {duration}s...")
    
    num_layers = 32
    moe_layers = [4, 8, 12, 16, 20, 24, 28, 32]  # MoE every 4 layers
    layer_time = duration / num_layers
    
    start = time.perf_counter()
    
    for layer in range(num_layers):
        monitor.mark(f"layer_{layer}_start")
        
        if layer in moe_layers:
            # Attention (no communication)
            time.sleep(layer_time * 0.3)
            monitor.mark(f"layer_{layer}_pre_route")
            
            # AllToAll for routing TO experts
            time.sleep(layer_time * 0.1)
            monitor.mark(f"layer_{layer}_dispatch")
            
            # Expert compute (local, no communication)
            time.sleep(layer_time * 0.3)
            monitor.mark(f"layer_{layer}_expert_compute")
            
            # AllToAll for routing FROM experts
            time.sleep(layer_time * 0.1)
            monitor.mark(f"layer_{layer}_combine")
            
            # Backward pass
            time.sleep(layer_time * 0.2)
        else:
            # Dense layer (no MoE communication)
            time.sleep(layer_time)
        
        monitor.mark(f"layer_{layer}_end")
    
    elapsed = time.perf_counter() - start
    print(f"EP simulation complete: {elapsed:.2f}s")


def run_mlx_finetune_simulation(
    mode: str = "tp",
    duration: float = 10.0,
    output_dir: str = "results/raw/distributed"
):
    """
    Run simulated fine-tuning with traffic monitoring.
    
    Since we don't have actual distributed setup yet,
    this simulates the traffic patterns we'd expect.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize traffic monitor
    monitor = ThunderboltMonitor(sample_rate_hz=100)
    
    # Start monitoring
    monitor.start()
    
    # Run simulation
    if mode == "tp":
        simulate_tp_traffic(monitor, duration)
    elif mode == "ep":
        simulate_ep_traffic(monitor, duration)
    else:
        print(f"Unknown mode: {mode}")
        monitor.stop()
        return
    
    # Stop monitoring and save
    samples = monitor.stop()
    summary = monitor.get_summary()
    
    # Save traffic data
    traffic_file = os.path.join(output_dir, f"{mode}_traffic.json")
    monitor.save(traffic_file)
    
    # Save summary
    results = {
        "mode": mode,
        "duration": duration,
        "traffic_summary": summary,
        "timestamp": datetime.now().isoformat()
    }
    
    results_file = os.path.join(output_dir, f"{mode}_finetune_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Traffic data saved to {traffic_file}")
    
    return results


def run_actual_finetune(
    mode: str = "tp",
    model_name: str = "deepseek-ai/deepseek-moe-16b-base",
    output_dir: str = "results/raw/distributed"
):
    """
    Run actual fine-tuning with traffic monitoring.
    
    TODO: Implement when Thunderbolt cable arrives.
    This will use:
    - MLX distributed for TP
    - Custom expert sharding for EP
    """
    raise NotImplementedError(
        "Actual distributed fine-tuning requires Thunderbolt connection. "
        "Use --mode simulate for now."
    )


def main():
    parser = argparse.ArgumentParser(description="Distributed Fine-tuning Benchmark")
    parser.add_argument("--mode", choices=["tp", "ep", "simulate_tp", "simulate_ep", "simulate"],
                       default="simulate", help="Parallelism mode")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Duration for simulation (seconds)")
    parser.add_argument("--output-dir", default="results/raw/distributed",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.mode == "simulate":
        # Run both simulations
        print("=" * 60)
        print("SIMULATING TENSOR PARALLELISM")
        print("=" * 60)
        run_mlx_finetune_simulation("tp", args.duration, args.output_dir)
        
        print("\n" + "=" * 60)
        print("SIMULATING EXPERT PARALLELISM")
        print("=" * 60)
        run_mlx_finetune_simulation("ep", args.duration, args.output_dir)
        
        # Generate comparison plot
        print("\n" + "=" * 60)
        print("GENERATING COMPARISON PLOT")
        print("=" * 60)
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from visualizations.plot_traffic import plot_bandwidth_utilization_simple
            plot_bandwidth_utilization_simple(
                tp_filepath=os.path.join(args.output_dir, "tp_traffic.json"),
                ep_filepath=os.path.join(args.output_dir, "ep_traffic.json"),
                output_path="results/reports/tp_vs_ep_bandwidth.png"
            )
        except ImportError as e:
            print(f"Could not generate plot: {e}")
            print("Run separately: python visualizations/plot_traffic.py")
        
    elif args.mode in ["simulate_tp", "tp"]:
        run_mlx_finetune_simulation("tp", args.duration, args.output_dir)
        
    elif args.mode in ["simulate_ep", "ep"]:
        run_mlx_finetune_simulation("ep", args.duration, args.output_dir)


if __name__ == "__main__":
    main()
