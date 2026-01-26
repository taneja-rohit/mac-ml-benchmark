"""
Roofline Analysis: Consumer vs. Datacenter Accelerators
Including NVIDIA, AMD, Huawei, Groq, and Apple Silicon

Updated: January 26, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_datacenter_roofline():
    """
    Roofline comparison across consumer and datacenter accelerators.
    Shows how different architectures optimize for different workloads.
    """
    
    # Hardware specs: (Name, Peak TFLOPS, Bandwidth TB/s, Color, Marker)
    hardware = [
        # Consumer (Apple Silicon)
        ("M4 Pro", 6.1, 0.253, "#dc2626", "o"),           # Red
        ("M5", 13.8, 0.118, "#2563eb", "o"),              # Blue
        
        # China (Huawei)
        ("Ascend 910D", 2000, 5.4, "#facc15", "s"),       # Yellow square
        
        # Inference Specialist
        ("Groq LPU", 188, 80.0, "#8b5cf6", "D"),          # Purple diamond
        
        # Datacenter (NVIDIA)
        ("H100 SXM", 1979, 3.35, "#22c55e", "^"),         # Green triangle
        ("B200", 4500, 8.0, "#84cc16", "^"),              # Lime triangle
        ("Rubin", 10000, 10.0, "#f97316", "^"),           # Orange triangle
        ("Rubin CPX", 20000, 12.0, "#ef4444", "^"),       # Red triangle
        
        # AMD
        ("MI450", 12000, 19.6, "#ec4899", "p"),           # Pink pentagon
    ]
    
    # LLM workload intensities (FLOPs/Byte)
    workloads = {
        "Decode\n(1 tok)": 1.0,
        "Prefill\n(batch=8)": 16.0,
        "Training\n(batch=32)": 128.0,
        "Large Batch\nTraining": 512.0,
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    intensities = np.logspace(-1, 4, 500)
    
    for name, tflops, bw_tbs, color, marker in hardware:
        bw_gbs = bw_tbs * 1000  # Convert to GB/s
        ridge = tflops * 1000 / bw_gbs  # FLOPs/Byte
        
        # Roofline
        roofline = np.minimum(intensities * bw_gbs / 1000, tflops)
        ax.loglog(intensities, roofline, 
                  label=f'{name} (Ridge: {ridge:.0f})', 
                  color=color, linewidth=2.5)
        
        # Mark ridge point
        ax.scatter([ridge], [tflops], color=color, s=120, zorder=5, 
                   edgecolors='black', linewidths=1, marker=marker)
    
    # Add workload markers
    for workload_name, intensity in workloads.items():
        ax.axvline(x=intensity, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.text(intensity * 1.1, 0.15, workload_name, rotation=0, va='bottom', 
                fontsize=9, color='#374151', fontweight='bold')
    
    # Styling
    ax.set_xlim(0.1, 5000)
    ax.set_ylim(0.1, 30000)
    ax.set_xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance (TFLOPS)', fontsize=13, fontweight='bold')
    ax.set_title('2026 Accelerator Roofline: The Ridge Point Defines the Workload\n"Left = Inference, Right = Training"', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add zone annotations with boxes
    ax.annotate('INFERENCE ZONE\n(Memory-Bound)', xy=(2, 0.8), fontsize=12, fontweight='bold', 
                ha='center', color='#8b5cf6',
                bbox=dict(boxstyle='round', facecolor='#f3e8ff', alpha=0.8))
    ax.annotate('TRAINING ZONE\n(Compute-Bound)', xy=(800, 15000), fontsize=12, fontweight='bold',
                ha='center', color='#ea580c',
                bbox=dict(boxstyle='round', facecolor='#fff7ed', alpha=0.8))
    
    # Groq callout
    ax.annotate('Groq: Ridge=2.4\n(Inference King)', 
                xy=(2.4, 188), xytext=(8, 50),
                arrowprops=dict(arrowstyle='->', color='#8b5cf6', lw=1.5),
                fontsize=9, color='#8b5cf6', fontweight='bold')
    
    # M5 callout
    ax.annotate('M5: Ridge=117\n(Inference Starver)', 
                xy=(117, 13.8), xytext=(300, 5),
                arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.5),
                fontsize=9, color='#2563eb', fontweight='bold')
    
    # Save
    os.makedirs('results/reports', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/reports/roofline_datacenter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Datacenter roofline saved to results/reports/roofline_datacenter.png")

if __name__ == "__main__":
    plot_datacenter_roofline()
