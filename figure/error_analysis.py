import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib.ticker import MaxNLocator

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

# Set random seed for reproducibility
np.random.seed(42)

# Account types
account_types = ['Exchange', 'ICO-Wallet', 'Mining', 'Phish/Hack', 'DeFi', 'Bridge']

# Create normalized confusion matrix from the paper data
conf_matrix = np.array([
    [98.76, 0.28, 0.14, 0.38, 0.32, 0.12],
    [0.41, 97.32, 0.21, 0.87, 0.73, 0.46],
    [0.27, 0.19, 98.63, 0.42, 0.31, 0.18],
    [0.33, 0.54, 0.28, 97.85, 0.63, 0.37],
    [0.57, 0.78, 0.34, 0.62, 96.83, 0.86],
    [0.32, 0.87, 0.21, 0.73, 1.89, 95.98]
])

# INCREASE FIGURE SIZE for better spacing
fig = plt.figure(figsize=(8.0, 6.0), dpi=300)  # Increased from 7.2x5.4
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# (a) Normalized confusion matrix
ax1 = plt.subplot(gs[0, 0])

# Create heatmap with custom colormap
cmap = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap=cmap, 
            xticklabels=account_types, yticklabels=account_types, 
            cbar_kws={'label': 'Percentage (%)'}, ax=ax1)

# Highlight the most challenging distinction pairs
ax1.add_patch(patches.Rectangle((4, 5), 1, 1, fill=False, edgecolor='red', lw=2))
ax1.add_patch(patches.Rectangle((3, 1), 1, 1, fill=False, edgecolor='red', lw=1.5))

ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
ax1.set_title('(a) Normalized Confusion Matrix (%)', fontweight='bold')
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

# (b) Transaction pattern visualization for challenging cases
ax2 = plt.subplot(gs[0, 1])

# Generate synthetic transaction patterns
days = np.arange(30)

# Create typical patterns for different account types
bridge_pattern = np.zeros(30)
bridge_pattern[5:25] = np.random.normal(10, 2, 20)
bridge_pattern[15] = 25

defi_pattern = np.zeros(30)
defi_pattern[5:25] = np.random.normal(8, 2, 20)
defi_pattern[10] = 22
defi_pattern[20] = 20

ico_pattern = np.zeros(30)
ico_pattern[10:15] = np.random.normal(30, 5, 5)

phishing_pattern = np.zeros(30)
phishing_pattern[12:16] = np.random.normal(25, 8, 4)

# The challenging cases (similar patterns)
bridge_as_defi = np.zeros(30)
bridge_as_defi[5:25] = np.random.normal(9, 2, 20)
bridge_as_defi[10] = 20
bridge_as_defi[15] = 22
bridge_as_defi[20] = 18

ico_as_phishing = np.zeros(30)
ico_as_phishing[12:17] = np.random.normal(27, 6, 5)

# Plot the patterns
ax2.plot(days, bridge_pattern, color='blue', linestyle='-', label='Typical Bridge', alpha=0.5)
ax2.plot(days, defi_pattern, color='green', linestyle='-', label='Typical DeFi', alpha=0.5)
ax2.plot(days, bridge_as_defi, color='red', linestyle='--', 
         label='Bridge misclassified as DeFi', linewidth=2)

# IMPROVED INSET POSITIONING AND SIZE
ax2_inset = ax2.inset_axes([0.58, 0.58, 0.40, 0.38])  # Adjusted position
ax2_inset.plot(days, ico_pattern, color='purple', linestyle='-', label='Typical ICO-Wallet', alpha=0.5)
ax2_inset.plot(days, phishing_pattern, color='orange', linestyle='-', label='Typical Phishing', alpha=0.5)
ax2_inset.plot(days, ico_as_phishing, color='red', linestyle='--', 
                label='ICO misclassified as Phishing', linewidth=2)
ax2_inset.set_title('ICO-Wallet vs Phishing', fontsize=7)
ax2_inset.set_xlabel('Days', fontsize=6)
ax2_inset.set_ylabel('Activity', fontsize=6)
ax2_inset.tick_params(axis='both', which='major', labelsize=6)
# SIMPLIFIED LEGEND with fewer items
ax2_inset.legend(fontsize=5, loc='upper right', framealpha=0.9, 
                ncol=1)  # Simplified to 1 column

ax2.set_xlabel('Days')
ax2.set_ylabel('Transaction Activity')
ax2.set_title('(b) Transaction Pattern Visualization', fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.7)

# (c) Attention weight visualization - COMPLETELY REDESIGNED
ax3 = plt.subplot(gs[1, 0])

# SIMPLIFIED - Use fewer features to reduce clutter
features = ['Tx Count', 'Avg Value', 'Burst Pattern', 'Contract Calls', 'Reciprocity']

# SIMPLER DISPLAY with fewer numbers
# Concentrated weights for correct classification
correct_weights = np.zeros((2, len(features)))
correct_weights[0, :] = [0.05, 0.08, 0.35, 0.25, 0.10]  # Bridge
correct_weights[1, :] = [0.06, 0.08, 0.04, 0.45, 0.15]  # DeFi

# More diffuse weights for incorrect classification
incorrect_weights = np.zeros((2, len(features)))
incorrect_weights[0, :] = [0.12, 0.14, 0.15, 0.17, 0.12]  # Bridge misclassified
incorrect_weights[1, :] = [0.10, 0.12, 0.12, 0.18, 0.14]  # DeFi misclassified

# BETTER POSITIONING of heatmaps
ax3.text(0.5, 0.96, '(c) Attention Weight Visualization', fontweight='bold', 
        ha='center', va='center', transform=ax3.transAxes)

# INCREASED SPACING between the two heatmaps
ax3_left = ax3.inset_axes([0.05, 0.15, 0.43, 0.7])  # More space to left edge
ax3_right = ax3.inset_axes([0.58, 0.15, 0.43, 0.7])  # More space between heatmaps

# REDUCE ANNOTATION SIZE and simplify
sns.heatmap(correct_weights, annot=True, fmt='.2f', cmap='Blues', 
            cbar=False, ax=ax3_left, vmin=0, vmax=0.5,  # Removed colorbar
            annot_kws={"size": 5}, xticklabels=features, 
            yticklabels=['Bridge\n(Correct)', 'DeFi\n(Correct)'])
ax3_left.set_title('Concentrated Weights (Correct)', fontsize=8)
plt.setp(ax3_left.get_xticklabels(), rotation=45, ha='right', fontsize=6)
plt.setp(ax3_left.get_yticklabels(), rotation=0, fontsize=7)

# SINGLE COLORBAR for both heatmaps
sns.heatmap(incorrect_weights, annot=True, fmt='.2f', cmap='Reds', 
            cbar_kws={'label': 'Attention Weight'}, ax=ax3_right, vmin=0, vmax=0.5,
            annot_kws={"size": 5}, xticklabels=features,  # Reduced size
            yticklabels=['Bridge\n(Misclassified)', 'DeFi\n(Misclassified)'])
ax3_right.set_title('Diffuse Weights (Incorrect)', fontsize=8)
plt.setp(ax3_right.get_xticklabels(), rotation=45, ha='right', fontsize=6)
plt.setp(ax3_right.get_yticklabels(), rotation=0, fontsize=7)

# Remove main axis elements
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.set_xticks([])
ax3.set_yticks([])

# (d) Feature importance analysis - SIMPLIFIED APPROACH
ax4 = plt.subplot(gs[1, 1])

# REDUCED NUMBER of features to improve readability
features = ['Tx Count', 'Send Value', 'Receive Value', 'Contract Calls', 'Burst Pattern']
importance = {
    'Exchange': [0.147, 0.135, 0.129, 0.121, 0.112],
    'ICO-Wallet': [0.128, 0.142, 0.124, 0.105, 0.165],
    'Mining': [0.138, 0.152, 0.132, 0.095, 0.108],
    'Phish/Hack': [0.125, 0.129, 0.121, 0.115, 0.162],
    'DeFi': [0.118, 0.125, 0.131, 0.168, 0.115],
    'Bridge': [0.122, 0.127, 0.118, 0.158, 0.102]
}

# IMPROVED BAR CHART SPACING
x = np.arange(len(account_types))
width = 0.15  # Wider bars with fewer features
multiplier = 0
feature_bars = []

# Create a bar for each feature
for feature_idx, feature in enumerate(features):
    feature_importance = [importance[acc_type][feature_idx] for acc_type in account_types]
    offset = width * multiplier
    feature_bar = ax4.bar(x + offset, feature_importance, width, label=feature)
    feature_bars.append(feature_bar)
    multiplier += 1

# SIMPLIFIED ANNOTATIONS - only annotate highest value per account type
for acc_idx, acc_type in enumerate(account_types):
    acc_importance = importance[acc_type]
    max_feature_idx = np.argmax(acc_importance)
    max_feature = features[max_feature_idx]
    max_value = acc_importance[max_feature_idx]
    
    # ROTATED TEXT for better visibility
    ax4.annotate(f"{max_feature}",
                xy=(acc_idx + width * max_feature_idx, max_value),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=6, rotation=45)

# Customize plot
ax4.set_ylabel('Normalized Importance')
ax4.set_title('(d) Feature Importance Analysis', fontweight='bold')
ax4.set_xticks(x + width * 2)  # Center ticks with fewer bars
ax4.set_xticklabels(account_types, rotation=30, ha='right')
ax4.yaxis.set_major_locator(MaxNLocator(5))
ax4.set_ylim(0, 0.2)
ax4.grid(axis='y', linestyle='--', alpha=0.7)

# SIMPLIFIED LEGEND with fewer items, placed outside the plot
ax4.legend(title="Features", loc='upper center', bbox_to_anchor=(0.5, -0.2), 
          ncol=3, frameon=True, fontsize=7)  # 3 columns instead of 4

# INCREASE SPACING between subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.40, wspace=0.30, bottom=0.2)

# Save figure
plt.savefig('error_analysis_improved.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('error_analysis_improved.png', format='png', bbox_inches='tight', dpi=300)

print("Improved figure created and saved as 'error_analysis_improved.pdf' and 'error_analysis_improved.png'")