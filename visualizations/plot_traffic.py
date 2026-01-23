"""
Traffic Visualization for Distributed Benchmarks

Creates clean bandwidth utilization plots comparing
Tensor Parallelism vs Expert Parallelism.

Usage:
    python visualizations/plot_traffic.py
"""

import json
import os
from typing import List, Optional
from dataclasses import dataclass

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not installed. Install with: pip install matplotlib")


@dataclass
class TrafficData:
    """Container for traffic data."""
    name: str
    timestamps: List[float]
    bandwidth_gbs: List[float]
    markers: List[dict]
    metadata: dict


def load_traffic_data(filepath: str, name: str = "Unknown") -> Optional[TrafficData]:
    """Load traffic data from JSON file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    with open(filepath) as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    
    return TrafficData(
        name=name,
        timestamps=[s["timestamp"] for s in samples],
        bandwidth_gbs=[s["total_rate_gbs"] for s in samples],
        markers=data.get("markers", []),
        metadata=data.get("metadata", {})
    )


def plot_single_traffic(data: TrafficData, output_path: str, title: str = None):
    """
    Plot bandwidth utilization over time for a single run.
    
    Simple, clean visualization showing:
    - Bandwidth (GB/s) on Y-axis
    - Time (s) on X-axis
    - Thunderbolt 4 max line (5 GB/s)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot bandwidth
    ax.fill_between(data.timestamps, data.bandwidth_gbs, alpha=0.3, color='#2563eb')
    ax.plot(data.timestamps, data.bandwidth_gbs, color='#2563eb', linewidth=1)
    
    # Thunderbolt 4 max line
    ax.axhline(y=5.0, color='#dc2626', linestyle='--', linewidth=1, label='TB4 Max (5 GB/s)')
    
    # Add markers
    for marker in data.markers:
        ax.axvline(x=marker["timestamp"], color='#6b7280', linestyle=':', alpha=0.5)
        ax.annotate(marker["label"], 
                    xy=(marker["timestamp"], ax.get_ylim()[1] * 0.9),
                    fontsize=8, rotation=90, va='top')
    
    # Styling
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=10)
    ax.set_title(title or f"Thunderbolt Traffic: {data.name}", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Stats annotation
    avg_bw = sum(data.bandwidth_gbs) / len(data.bandwidth_gbs) if data.bandwidth_gbs else 0
    max_bw = max(data.bandwidth_gbs) if data.bandwidth_gbs else 0
    stats_text = f"Avg: {avg_bw:.2f} GB/s | Peak: {max_bw:.2f} GB/s | Util: {(max_bw/5)*100:.0f}%"
    ax.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")


def plot_comparison(tp_data: TrafficData, ep_data: TrafficData, output_path: str):
    """
    Compare Tensor Parallelism vs Expert Parallelism traffic patterns.
    
    Two stacked plots showing the different patterns:
    - TP: Constant high bandwidth
    - EP: Bursty traffic
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # --- Tensor Parallelism ---
    ax1.fill_between(tp_data.timestamps, tp_data.bandwidth_gbs, alpha=0.3, color='#2563eb')
    ax1.plot(tp_data.timestamps, tp_data.bandwidth_gbs, color='#2563eb', linewidth=1)
    ax1.axhline(y=5.0, color='#dc2626', linestyle='--', linewidth=1)
    ax1.set_ylabel("Bandwidth (GB/s)", fontsize=10)
    ax1.set_title("Tensor Parallelism (AllReduce every layer)", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 6)
    ax1.grid(True, alpha=0.3)
    
    # TP stats
    tp_avg = sum(tp_data.bandwidth_gbs) / len(tp_data.bandwidth_gbs) if tp_data.bandwidth_gbs else 0
    tp_max = max(tp_data.bandwidth_gbs) if tp_data.bandwidth_gbs else 0
    ax1.annotate(f"Avg: {tp_avg:.2f} GB/s | Peak: {tp_max:.2f} GB/s", 
                 xy=(0.02, 0.85), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- Expert Parallelism ---
    ax2.fill_between(ep_data.timestamps, ep_data.bandwidth_gbs, alpha=0.3, color='#16a34a')
    ax2.plot(ep_data.timestamps, ep_data.bandwidth_gbs, color='#16a34a', linewidth=1)
    ax2.axhline(y=5.0, color='#dc2626', linestyle='--', linewidth=1, label='TB4 Max')
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Bandwidth (GB/s)", fontsize=10)
    ax2.set_title("Expert Parallelism (AllToAll during routing)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 6)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # EP stats
    ep_avg = sum(ep_data.bandwidth_gbs) / len(ep_data.bandwidth_gbs) if ep_data.bandwidth_gbs else 0
    ep_max = max(ep_data.bandwidth_gbs) if ep_data.bandwidth_gbs else 0
    ax2.annotate(f"Avg: {ep_avg:.2f} GB/s | Peak: {ep_max:.2f} GB/s", 
                 xy=(0.02, 0.85), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    fig.suptitle("Thunderbolt Traffic: Tensor Parallelism vs Expert Parallelism", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {output_path}")


def plot_bandwidth_utilization_simple(
    tp_filepath: str = None,
    ep_filepath: str = None,
    output_path: str = "results/reports/bandwidth_utilization.png"
):
    """
    Simple bandwidth utilization plot.
    
    If both TP and EP data exist, creates comparison.
    Otherwise plots whatever is available.
    """
    tp_data = load_traffic_data(tp_filepath, "Tensor Parallelism") if tp_filepath else None
    ep_data = load_traffic_data(ep_filepath, "Expert Parallelism") if ep_filepath else None
    
    if tp_data and ep_data:
        plot_comparison(tp_data, ep_data, output_path)
    elif tp_data:
        plot_single_traffic(tp_data, output_path, "Tensor Parallelism - Bandwidth Utilization")
    elif ep_data:
        plot_single_traffic(ep_data, output_path, "Expert Parallelism - Bandwidth Utilization")
    else:
        print("No traffic data found to plot")


def generate_sample_data():
    """Generate sample data for testing visualization."""
    import random
    import math
    
    os.makedirs("results/raw/distributed", exist_ok=True)
    
    # Simulated TP traffic (constant high)
    tp_samples = []
    for i in range(500):
        t = i * 0.01
        # Constant ~3.5 GB/s with small noise
        bw = 3.5 + random.gauss(0, 0.2)
        tp_samples.append({
            "timestamp": t,
            "total_rate_gbs": max(0, bw),
            "send_rate_gbs": max(0, bw/2),
            "recv_rate_gbs": max(0, bw/2)
        })
    
    with open("results/raw/distributed/tp_traffic_sample.json", "w") as f:
        json.dump({"metadata": {"name": "TP Sample"}, "markers": [], "samples": tp_samples}, f)
    
    # Simulated EP traffic (bursty)
    ep_samples = []
    for i in range(500):
        t = i * 0.01
        # Bursty: low during compute, high during routing
        phase = (t % 0.5) / 0.5  # 0-1 within each 0.5s cycle
        if phase < 0.7:  # 70% compute (low traffic)
            bw = 0.3 + random.gauss(0, 0.1)
        else:  # 30% routing (high traffic)
            bw = 4.5 + random.gauss(0, 0.3)
        ep_samples.append({
            "timestamp": t,
            "total_rate_gbs": max(0, bw),
            "send_rate_gbs": max(0, bw/2),
            "recv_rate_gbs": max(0, bw/2)
        })
    
    with open("results/raw/distributed/ep_traffic_sample.json", "w") as f:
        json.dump({"metadata": {"name": "EP Sample"}, "markers": [], "samples": ep_samples}, f)
    
    print("Sample data generated in results/raw/distributed/")


if __name__ == "__main__":
    # Generate sample data for testing
    generate_sample_data()
    
    # Plot comparison
    plot_bandwidth_utilization_simple(
        tp_filepath="results/raw/distributed/tp_traffic_sample.json",
        ep_filepath="results/raw/distributed/ep_traffic_sample.json",
        output_path="results/reports/bandwidth_utilization_sample.png"
    )
