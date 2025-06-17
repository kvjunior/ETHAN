import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import matplotlib.gridspec as gridspec
from scipy import stats

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

# Define colors that will be consistent across subplots
colors = plt.cm.tab10.colors
method_colors = {
    'Uncalibrated': '#d62728',  # Red
    'Temperature': '#ff7f0e',   # Orange
    'Vector': '#2ca02c',        # Green
    'Matrix': '#9467bd',        # Purple
    'Beta': '#8c564b',          # Brown
    'Histogram': '#e377c2',     # Pink
    'Isotonic': '#7f7f7f',      # Gray
    'Ensemble': '#1f77b4'       # Blue
}

# Define account types and calibration methods
account_types = ['Exchange', 'ICO-Wallet', 'Mining', 'Phish/Hack', 'DeFi', 'Bridge']
calibration_methods = ['Uncalibrated', 'Temperature', 'Vector', 'Matrix', 'Beta', 'Histogram', 'Isotonic', 'Ensemble']

# ECE values as mentioned in the paper
ece_values = {
    'Uncalibrated': [0.0873, 0.1145, 0.0967, 0.0912, 0.1293, 0.1432],
    'Temperature': [0.0341, 0.0475, 0.0382, 0.0371, 0.0532, 0.0587],
    'Vector': [0.0312, 0.0418, 0.0353, 0.0329, 0.0485, 0.0549],
    'Matrix': [0.0294, 0.0392, 0.0331, 0.0318, 0.0451, 0.0512],
    'Beta': [0.0285, 0.0374, 0.0312, 0.0297, 0.0423, 0.0478],
    'Histogram': [0.0272, 0.0362, 0.0298, 0.0285, 0.0408, 0.0465],
    'Isotonic': [0.0265, 0.0352, 0.0288, 0.0274, 0.0395, 0.0452],
    'Ensemble': [0.0187, 0.0243, 0.0219, 0.0203, 0.0287, 0.0315]
}

# Adaptive weights based on the paper
adaptive_weights = {
    'Exchange': {'Temperature': 0.142, 'Vector': 0.158, 'Matrix': 0.183, 'Beta': 0.192, 'Histogram': 0.165, 'Isotonic': 0.160},
    'ICO-Wallet': {'Temperature': 0.135, 'Vector': 0.152, 'Matrix': 0.169, 'Beta': 0.178, 'Histogram': 0.173, 'Isotonic': 0.193},
    'Mining': {'Temperature': 0.139, 'Vector': 0.155, 'Matrix': 0.175, 'Beta': 0.184, 'Histogram': 0.170, 'Isotonic': 0.177},
    'Phish/Hack': {'Temperature': 0.137, 'Vector': 0.153, 'Matrix': 0.172, 'Beta': 0.181, 'Histogram': 0.171, 'Isotonic': 0.186},
    'DeFi': {'Temperature': 0.129, 'Vector': 0.145, 'Matrix': 0.162, 'Beta': 0.171, 'Histogram': 0.186, 'Isotonic': 0.207},
    'Bridge': {'Temperature': 0.121, 'Vector': 0.137, 'Matrix': 0.162, 'Beta': 0.172, 'Histogram': 0.183, 'Isotonic': 0.225}
}

# Create figure with custom grid
fig = plt.figure(figsize=(7.2, 5.4), dpi=300)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# (a) Reliability diagrams - comparing before and after calibration
ax1 = plt.subplot(gs[0, 0])

# Generate synthetic reliability diagram data
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Create reliability curves (synthetic data based on paper description)
# Perfect calibration would be identity line
perfect_calibration = bin_centers

# Uncalibrated model (overconfident) - below the identity line
uncalibrated = bin_centers - 0.15 * np.sin(np.pi * bin_centers)

# Ensemble calibration - closer to identity line
ensemble = bin_centers - 0.02 * np.sin(np.pi * bin_centers)

# Plot reliability curves
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')
ax1.plot(bin_centers, uncalibrated, 'o-', color=method_colors['Uncalibrated'], 
         label='Uncalibrated (ECE=0.1104)', lw=1.5, markersize=4)
ax1.plot(bin_centers, ensemble, 'o-', color=method_colors['Ensemble'], 
         label='Ensemble (ECE=0.0242)', lw=1.5, markersize=4)

# Add gap visualization
for i, x in enumerate(bin_centers):
    ax1.plot([x, x], [uncalibrated[i], x], 'r-', alpha=0.3, lw=1)
    ax1.plot([x, x], [ensemble[i], x], 'b-', alpha=0.3, lw=1)

# Customize plot
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Empirical Accuracy')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_title('(a) Reliability Diagrams', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower right', frameon=True, framealpha=0.9)

# (b) Expected calibration error (ECE) for different methods
ax2 = plt.subplot(gs[0, 1])

# Organize ECE values for plotting
methods_to_plot = ['Uncalibrated', 'Temperature', 'Vector', 'Matrix', 'Beta', 'Histogram', 'Isotonic', 'Ensemble']
method_positions = np.arange(len(methods_to_plot))
average_ece = [np.mean(ece_values[method]) for method in methods_to_plot]

# Create barplot of average ECE
bars = ax2.bar(method_positions, average_ece, width=0.7, 
         color=[method_colors[m] for m in methods_to_plot], edgecolor='black', linewidth=0.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', rotation=90, fontsize=6)

# Customize plot
ax2.set_xticks(method_positions)
ax2.set_xticklabels([m if m != 'Uncalibrated' else 'Uncal.' for m in methods_to_plot], rotation=45, ha='right')
ax2.set_ylabel('Expected Calibration Error (ECE)')
ax2.set_title('(b) Expected Calibration Error', fontweight='bold')
ax2.set_ylim(0, 0.12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# (c) Adaptive weight distribution visualization
ax3 = plt.subplot(gs[1, 0])

# Prepare data for stacked bar chart
methods_for_weights = ['Temperature', 'Vector', 'Matrix', 'Beta', 'Histogram', 'Isotonic']
weights_data = np.array([[adaptive_weights[acc][method] for method in methods_for_weights] for acc in account_types])

# Create stacked bar chart
bottom = np.zeros(len(account_types))
for i, method in enumerate(methods_for_weights):
    values = weights_data[:, i]
    ax3.bar(account_types, values, bottom=bottom, label=method, 
            color=method_colors[method], edgecolor='black', linewidth=0.5)
    bottom += values

# Customize plot
ax3.set_ylabel('Weight in Ensemble')
ax3.set_title('(c) Adaptive Calibration Weights', fontweight='bold')
ax3.set_ylim(0, 1.05)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.legend(loc='upper right', frameon=True, framealpha=0.9)
plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')

# (d) Predicted probability distributions
ax4 = plt.subplot(gs[1, 1])

# Generate synthetic data for probability distributions
np.random.seed(42)  # For reproducibility

# Before calibration
n_samples = 1000
correct_before = np.random.beta(8, 2, n_samples)  # High confidence, correct
incorrect_before = np.random.beta(5, 2, n_samples)  # Also high confidence, but incorrect

# After calibration - better separation
correct_after = np.random.beta(10, 2, n_samples)  # Higher confidence for correct
incorrect_after = np.random.beta(2, 5, n_samples)  # Lower confidence for incorrect

# Create kernel density plots
sns.kdeplot(correct_before, ax=ax4, color='blue', linestyle='-', 
            label='Correct (Before)', fill=True, alpha=0.2)
sns.kdeplot(incorrect_before, ax=ax4, color='red', linestyle='-', 
            label='Incorrect (Before)', fill=True, alpha=0.2)
sns.kdeplot(correct_after, ax=ax4, color='blue', linestyle='--', 
            label='Correct (After)', lw=2)
sns.kdeplot(incorrect_after, ax=ax4, color='red', linestyle='--', 
            label='Incorrect (After)', lw=2)

# Customize plot
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Density')
ax4.set_title('(d) Probability Distributions', fontweight='bold')
ax4.set_xlim(0, 1)
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.legend(loc='upper center', frameon=True, framealpha=0.9)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Save figure
plt.savefig('calibration_analysis.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('calibration_analysis.png', format='png', bbox_inches='tight', dpi=300)
plt.close()

print("Figure created and saved as 'calibration_analysis.pdf' and 'calibration_analysis.png'")