import matplotlib.pyplot as plt
import numpy as np
import os

def plot_roofline():
    # Performance Data (Achieved from benchmarks)
    m5_peak_tflops = 13.84
    m5_bw_gbs = 118.4
    
    m4_pro_peak_tflops = 6.07
    m4_pro_bw_gbs = 253.2

    # Model Data: Mistral-7B FP16
    # Arithmetic Intensity = FLOPs / Byte
    # For inference (linear layers): ~2 FLOPs per 2 Bytes (weight) = 1.0 FLOP/Byte
    model_intensity = 1.0
    
    # Calculate Ridge Points (TFLOPS / BW)
    m5_ridge = m5_peak_tflops * 1000 / m5_bw_gbs
    m4_pro_ridge = m4_pro_peak_tflops * 1000 / m4_pro_bw_gbs
    
    # Intensity range for plot (log scale)
    intensities = np.logspace(-1, 3, 100)
    
    # Calculate Rooflines
    m5_roofline = np.minimum(intensities * m5_bw_gbs / 1000, m5_peak_tflops)
    m4_pro_roofline = np.minimum(intensities * m4_pro_bw_gbs / 1000, m4_pro_peak_tflops)
    
    # Create Plot
    plt.figure(figsize=(12, 8))
    
    # Plot M5 Roofline
    plt.loglog(intensities, m5_roofline, label=f'M5 Roofline (Peak: {m5_peak_tflops} TFLOPS, BW: {m5_bw_gbs} GB/s)', 
               color='#2563eb', linewidth=2)
    plt.axvline(x=m5_ridge, color='#2563eb', linestyle='--', alpha=0.3)
    
    # Plot M4 Pro Roofline
    plt.loglog(intensities, m4_pro_roofline, label=f'M4 Pro Roofline (Peak: {m4_pro_peak_tflops} TFLOPS, BW: {m4_pro_bw_gbs} GB/s)', 
               color='#dc2626', linewidth=2)
    plt.axvline(x=m4_pro_ridge, color='#dc2626', linestyle='--', alpha=0.3)
    
    # Plot Mistral-7B point
    m5_achieved_inf = (model_intensity * m5_bw_gbs / 1000) # In memory-bound region
    m4_achieved_inf = (model_intensity * m4_pro_bw_gbs / 1000) # In memory-bound region
    
    plt.scatter([model_intensity], [m5_achieved_inf], color='#2563eb', s=100, zorder=5, label='Mistral-7B on M5')
    plt.scatter([model_intensity], [m4_achieved_inf], color='#dc2626', s=100, zorder=5, label='Mistral-7B on M4 Pro')
    
    # Add explanatory arrows/text
    plt.annotate('Memory Bound Zone', xy=(0.2, 0.1), xytext=(0.2, 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=12, fontweight='bold')
    
    plt.annotate('Compute Bound Zone', xy=(100, 1), xytext=(100, 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=12, fontweight='bold')

    plt.annotate('Mistral-7B Intensity (1.0)', xy=(model_intensity, 0.01), xytext=(model_intensity, 0.005),
                 ha='center', fontsize=10, color='purple')

    # Styling
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=12)
    plt.ylabel('Performance (TFLOPS)', fontsize=12)
    plt.title('M5 vs M4 Pro: Roofline Analysis (Mistral-7B Inference)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    
    # Annotate Ridge Points
    plt.text(m5_ridge, 20, f'  M5 Ridge: {m5_ridge:.1f}', color='#2563eb', fontsize=9, va='bottom')
    plt.text(m4_pro_ridge, 20, f'  M4 Ridge: {m4_pro_ridge:.1f}', color='#dc2626', fontsize=9, va='bottom')

    # Save
    os.makedirs('results/reports', exist_ok=True)
    plt.savefig('results/reports/roofline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Roofline plot saved to results/reports/roofline_comparison.png")

if __name__ == "__main__":
    plot_roofline()
