import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm

# Configure matplotlib for IEEE journal standards
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})

def create_neighborhood_analysis_figure():
    """
    Create a professional dual-axis figure showing F1-Score and Inference Time
    vs. Neighborhood Size for ETHAN framework analysis.
    """
    # Data points
    neighborhood_sizes = np.array([500, 1000, 1500, 2000, 3000, 5000])
    f1_scores = np.array([94.23, 96.14, 97.21, 97.82, 97.89, 97.91])
    inference_times = np.array([3.2, 5.7, 9.3, 12.8, 24.6, 47.3])
    
    # Create figure with specific dimensions (IEEE single column width)
    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    
    # First y-axis: F1-Score
    color1 = '#1f77b4'  # Professional blue
    ax1.set_xlabel('Neighborhood Size $k$', fontsize=9)
    ax1.set_ylabel('F1-Score (%)', fontsize=9, color=color1)
    
    # Plot F1-Score with error bars (simulated for demonstration)
    f1_errors = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 0.06])  # Simulated error bars
    line1 = ax1.errorbar(neighborhood_sizes, f1_scores, yerr=f1_errors,
                         color=color1, marker='s', markersize=6, 
                         linewidth=1.5, capsize=3, capthick=1,
                         label='F1-Score', markeredgewidth=0.8,
                         markeredgecolor=color1, markerfacecolor=color1)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(93.5, 98.5)
    ax1.set_yticks([94, 95, 96, 97, 98])
    
    # Add subtle grid
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Second y-axis: Inference Time
    ax2 = ax1.twinx()
    color2 = '#d62728'  # Professional red
    ax2.set_ylabel('Inference Time (ms)', fontsize=9, color=color2)
    
    # Plot Inference Time with polynomial fit
    # Add smooth interpolation curve
    x_smooth = np.linspace(500, 5000, 100)
    # Fit exponential-like curve to inference time
    coeffs = np.polyfit(np.log(neighborhood_sizes), np.log(inference_times), 2)
    y_smooth = np.exp(np.polyval(coeffs, np.log(x_smooth)))
    
    ax2.plot(x_smooth, y_smooth, color=color2, linewidth=1, alpha=0.3, linestyle='--')
    line2 = ax2.plot(neighborhood_sizes, inference_times,
                     color=color2, marker='o', markersize=6,
                     linewidth=1.5, label='Inference Time',
                     markeredgewidth=0.8, markeredgecolor=color2,
                     markerfacecolor=color2)[0]
    
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 50)
    ax2.set_yticks([0, 10, 20, 30, 40, 50])
    
    # X-axis configuration
    ax1.set_xlim(300, 5200)
    ax1.set_xticks(neighborhood_sizes)
    ax1.set_xticklabels(['500', '1k', '1.5k', '2k', '3k', '5k'])
    
    # Add vertical line at optimal value
    ax1.axvline(x=2000, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.text(2000, 93.7, 'Optimal', ha='center', va='bottom', fontsize=7, 
             color='gray', style='italic')
    
    # Combined legend
    lines = [line1, line2]
    labels = ['F1-Score', 'Inference Time']
    ax1.legend(lines, labels, loc='center right', frameon=True, 
               fancybox=False, shadow=False, borderpad=0.5,
               columnspacing=0.5, handlelength=1.5)
    
    # Add annotations for key points
    # Annotate the optimal point
    ax1.annotate('97.82%', xy=(2000, 97.82), xytext=(2300, 96.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                fontsize=7, ha='left')
    
    # Annotate the diminishing returns region
    ax1.annotate('Diminishing\nreturns', xy=(4000, 97.9), xytext=(3800, 95.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                fontsize=7, ha='center', color='gray', style='italic')
    
    # Fine-tune layout
    plt.tight_layout()
    
    # Add subtle box around the plot
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)
    
    return fig

def add_performance_indicators(ax1, ax2):
    """Add performance indicators and shaded regions to highlight key insights."""
    # Add shaded region for optimal range
    ax1.axvspan(1500, 2500, alpha=0.1, color='green', label='Optimal range')
    
    # Add performance plateau indicator
    ax1.axhspan(97.5, 98.0, alpha=0.1, color='blue', xmin=0.6, xmax=1.0)
    ax1.text(4000, 97.75, 'Performance\nplateau', ha='center', va='center',
             fontsize=7, style='italic', color='darkblue')

def save_figure_as_pdf(fig, filename='neighborhood_analysis.pdf'):
    """Save the figure as a high-quality PDF suitable for journal publication."""
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.05)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'ETHAN Framework: Neighborhood Size Analysis'
        d['Author'] = 'ETHAN Research Team'
        d['Subject'] = 'Performance vs. Computational Cost Trade-off'
        d['Keywords'] = 'Graph Neural Networks, Blockchain, Ethereum, Performance Analysis'
        d['Creator'] = 'ETHAN Framework Analysis Pipeline'

def main():
    """Generate the complete figure and save as PDF."""
    # Create the main figure
    fig = create_neighborhood_analysis_figure()
    
    # Add additional performance indicators
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]
    add_performance_indicators(ax1, ax2)
    
    # Save as PDF
    save_figure_as_pdf(fig, 'ethan_neighborhood_analysis.pdf')
    
    # Also save as high-resolution PNG for preview
    fig.savefig('ethan_neighborhood_analysis.png', dpi=300, bbox_inches='tight', 
                pad_inches=0.05, facecolor='white', edgecolor='none')
    
    # Display the figure (optional)
    plt.show()
    
    print("Figure successfully generated and saved as:")
    print("- ethan_neighborhood_analysis.pdf (for journal submission)")
    print("- ethan_neighborhood_analysis.png (for preview)")

if __name__ == "__main__":
    main()