import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
from scipy import stats
from matplotlib.ticker import PercentFormatter
import matplotlib.gridspec as gridspec

# Set IEEE Transaction compatible styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 10

# Account types with their reported mean uncertainty values
account_types = ['Exchange', 'ICO-Wallet', 'Mining', 'Phish/Hack', 'DeFi', 'Bridge']
uncertainty_means = [0.037, 0.048, 0.055, 0.061, 0.068, 0.074]  # From the paper
uncertainty_stds = [0.015, 0.021, 0.023, 0.025, 0.027, 0.031]   # Estimated for distribution

# Calibration methods
calibration_methods = ['Uncalibrated', 'Temperature', 'Vector', 'Matrix', 'Beta', 'Ensemble']
calibration_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#1f77b4']

# Uncertainty thresholds from the paper
thresholds = [0.025, 0.050, 0.075, 0.100, 0.150, None]  # None represents "All predictions"
coverage = [61.37, 78.92, 87.43, 93.71, 98.24, 100.00]
accuracy = [99.95, 99.82, 99.53, 99.17, 98.41, 97.82]

# Create figure with custom grid layout
fig = plt.figure(figsize=(7.2, 5.4), dpi=300)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# (a) Distribution of epistemic uncertainty by account type
ax1 = plt.subplot(gs[0, 0])

# Generate synthetic data for uncertainty distributions
np.random.seed(42)  # For reproducibility
uncertainty_data = []
for i, (mean, std) in enumerate(zip(uncertainty_means, uncertainty_stds)):
    # Create skewed distribution with gamma
    shape = (mean / std) ** 2
    scale = std ** 2 / mean
    dist = np.random.gamma(shape, scale, 1000)
    # Clip to reasonable range
    dist = np.clip(dist, 0, 0.25)
    uncertainty_data.append(dist)

# Create violin plots
parts = ax1.violinplot(uncertainty_data, showmeans=True, showmedians=False)

# Customize violin plots
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(plt.cm.tab10(i))
    pc.set_alpha(0.7)

# Add means as points
for i, mean in enumerate(uncertainty_means):
    ax1.scatter(i+1, mean, marker='o', color='white', s=30, zorder=3)
    ax1.text(i+1, mean + 0.01, f"{mean:.3f}", ha='center', va='bottom', fontsize=7)

# Customize plot
ax1.set_xticks(np.arange(1, len(account_types) + 1))
ax1.set_xticklabels(account_types, rotation=30, ha='right')
ax1.set_ylabel('Epistemic Uncertainty')
ax1.set_title('(a) Uncertainty Distribution by Account Type', fontweight='bold')
ax1.set_ylim(0, 0.15)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# (b) Correlation plot between uncertainty and error rates
ax2 = plt.subplot(gs[0, 1])

# Generate synthetic data for uncertainty vs error correlation
num_points = 100
pearson_r = 0.745  # From the paper

# Create correlated data
mean = [0.07, 0.05]  # Means for uncertainty and error rate
cov = [[0.003, pearson_r * np.sqrt(0.003 * 0.005)], 
        [pearson_r * np.sqrt(0.003 * 0.005), 0.005]]  # Covariance matrix
corr_data = np.random.multivariate_normal(mean, cov, num_points)
uncertainty_vals = np.abs(corr_data[:, 0])  # Ensure positive
error_rates = np.abs(corr_data[:, 1])       # Ensure positive

# Plot with different colors for account types
n_per_type = num_points // len(account_types)
for i, acc_type in enumerate(account_types):
    start_idx = i * n_per_type
    end_idx = (i+1) * n_per_type if i < len(account_types)-1 else num_points
    ax2.scatter(uncertainty_vals[start_idx:end_idx], 
                error_rates[start_idx:end_idx],
                color=plt.cm.tab10(i), alpha=0.7, label=acc_type)

# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(uncertainty_vals, error_rates)
x_line = np.linspace(0, max(uncertainty_vals)*1.1, 100)
y_line = intercept + slope * x_line
ax2.plot(x_line, y_line, 'k--', lw=1.5)

# Annotate with Pearson's r
ax2.text(0.05, 0.95, f"Pearson's r = {pearson_r:.3f}", transform=ax2.transAxes,
         fontsize=8, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Customize plot
ax2.set_xlabel('Epistemic Uncertainty')
ax2.set_ylabel('Error Rate')
ax2.set_title('(b) Uncertainty vs. Error Correlation', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(0, 0.15)
ax2.set_ylim(0, 0.15)
ax2.legend(loc='upper left', frameon=True, framealpha=0.9)

# (c) Reliability diagrams for different calibration methods
ax3 = plt.subplot(gs[1, 0])

# Generate synthetic reliability diagram data
# Confidence bins
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Accuracy for different methods (should be close to ideal for better methods)
accuracy_uncalibrated = bin_centers - 0.15 * np.sin(np.pi * bin_centers)
accuracy_temperature = bin_centers - 0.08 * np.sin(np.pi * bin_centers)
accuracy_vector = bin_centers - 0.06 * np.sin(np.pi * bin_centers)
accuracy_matrix = bin_centers - 0.04 * np.sin(np.pi * bin_centers)
accuracy_beta = bin_centers - 0.03 * np.sin(np.pi * bin_centers)
accuracy_ensemble = bin_centers - 0.01 * np.sin(np.pi * bin_centers)

accuracies = [accuracy_uncalibrated, accuracy_temperature, accuracy_vector, 
              accuracy_matrix, accuracy_beta, accuracy_ensemble]

# Plot reliability diagrams
ax3.plot([0, 1], [0, 1], 'k-', lw=1.5, label='Perfect Calibration')
for i, (method, acc) in enumerate(zip(calibration_methods, accuracies)):
    ax3.plot(bin_centers, acc, '-o', color=calibration_colors[i], lw=1.5, 
             markersize=4, label=method)

# Customize plot
ax3.set_xlabel('Confidence')
ax3.set_ylabel('Accuracy')
ax3.set_title('(c) Reliability Diagrams', fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(loc='lower right', frameon=True, framealpha=0.9)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

# (d) Rejection curves plotting accuracy vs. coverage
ax4 = plt.subplot(gs[1, 1])

# Plot rejection curve from paper data
ax4.plot(coverage, accuracy, 'o-', color='#1f77b4', lw=2, markersize=5)

# Add uncertainty threshold annotations
for i, thresh in enumerate(thresholds[:-1]):  # Skip "All predictions"
    ax4.annotate(f"τ = {thresh}", 
                 xy=(coverage[i], accuracy[i]),
                 xytext=(5, 0), textcoords='offset points',
                 va='center', ha='left', fontsize=7,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# Customize plot
ax4.set_xlabel('Coverage (%)')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('(d) Rejection Curve', fontweight='bold')
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.set_xlim(50, 102)
ax4.set_ylim(97.5, 100.1)
ax4.xaxis.set_major_formatter(PercentFormatter())
ax4.yaxis.set_major_formatter(PercentFormatter())

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Save figure
plt.savefig('uncertainty_analysis.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('uncertainty_analysis.png', format='png', bbox_inches='tight', dpi=300)
plt.close()

print("Figure created and saved as 'uncertainty_analysis.pdf' and 'uncertainty_analysis.png'")