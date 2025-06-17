import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib import cm

# Set up IEEE-compatible styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 10

# Account types
account_types = ['Exchange', 'ICO-Wallet', 'Mining', 'Phish/Hack', 'DeFi', 'Bridge']
# Methods for comparison
methods = ['ETHAN (Ours)', 'BPA-GNN', 'TEGDetector', 'Ethident']

# Colors for consistent visualization
colors = plt.cm.tab10(np.linspace(0, 1, 6))
method_colors = plt.cm.Paired(np.linspace(0, 1, 4))

# Create figure with custom grid
fig = plt.figure(figsize=(7.2, 5.4), dpi=300)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# Generate synthetic data based on paper results
# F1 scores from the paper
f1_scores = {
    'ETHAN (Ours)': [99.46, 97.64, 98.32, 98.71, 96.83, 95.98],
    'BPA-GNN': [89.31, 84.15, 86.73, 85.14, 84.91, 83.62],
    'TEGDetector': [85.67, 80.77, 84.65, 80.86, 82.73, 80.95],
    'Ethident': [87.23, 70.97, 66.67, 88.93, 85.69, 84.25]
}

# Synthetic confusion matrix (using values from the paper)
conf_matrix = np.array([
    [98.76, 0.28, 0.14, 0.38, 0.32, 0.12],
    [0.41, 97.32, 0.21, 0.87, 0.73, 0.46],
    [0.27, 0.19, 98.63, 0.42, 0.31, 0.18],
    [0.33, 0.54, 0.28, 97.85, 0.63, 0.37],
    [0.57, 0.78, 0.34, 0.62, 96.83, 0.86],
    [0.32, 0.87, 0.21, 0.73, 1.89, 95.98]
])

# Synthetic data for precision-recall and ROC curves
def generate_curve_data(f1_baseline, auc_baseline):
    """Generate synthetic PR and ROC curve data based on F1 and AUC values"""
    n_points = 100
    # Better F1 typically means better precision and recall
    precision_factor = f1_baseline / 100
    recall_factor = f1_baseline / 100
    
    # Generate recall points
    recall = np.linspace(0, 1, n_points)
    
    # Generate precision with higher values for better F1
    precision = np.zeros_like(recall)
    for i, r in enumerate(recall):
        if r < 0.9:  # High precision for most of the curve
            p_base = 1 - (1-precision_factor) * (r ** (1.5 - precision_factor))
            precision[i] = min(1.0, p_base + 0.05 * np.random.randn() * precision_factor)
        else:  # Precision drops at high recall
            drop_factor = (r - 0.9) / 0.1
            p_base = precision[int(0.9*n_points)] * (1 - drop_factor * (1 - precision_factor))
            precision[i] = max(0.5, min(1.0, p_base + 0.02 * np.random.randn()))
    
    # Generate ROC curve data based on AUC
    auc_factor = auc_baseline / 100
    fpr = np.linspace(0, 1, n_points)
    tpr = np.zeros_like(fpr)
    
    # Generate TPR based on AUC
    for i, fp in enumerate(fpr):
        if auc_factor > 0.95:  # Very good AUC
            tpr[i] = min(1.0, 1 - (1-fp)**(10*auc_factor) + 0.01 * np.random.randn())
        else:  # Less ideal AUC
            tpr[i] = min(1.0, fp + auc_factor * (1 - fp) + 0.02 * np.random.randn())
    
    # Ensure curve starts at (0,0) and ends at (1,1)
    precision[0], recall[0] = 1.0, 0.0
    precision[-1] = recall[-1]
    tpr[0], fpr[0] = 0.0, 0.0
    tpr[-1], fpr[-1] = 1.0, 1.0
    
    return precision, recall, fpr, tpr

# Generate curve data for each account type
curve_data = {}
for i, acc_type in enumerate(account_types):
    f1 = f1_scores['ETHAN (Ours)'][i]
    # Estimate AUC based on the table in the paper
    if acc_type == 'Exchange': auc_val = 99.87
    elif acc_type == 'ICO-Wallet': auc_val = 99.21
    elif acc_type == 'Mining': auc_val = 99.56
    elif acc_type == 'Phish/Hack': auc_val = 99.63
    elif acc_type == 'DeFi': auc_val = 98.92
    else: auc_val = 98.47  # Bridge
    
    precision, recall, fpr, tpr = generate_curve_data(f1, auc_val)
    curve_data[acc_type] = {
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc_val/100
    }

# Plot (a): Precision-Recall curves
ax1 = plt.subplot(gs[0, 0])
for i, acc_type in enumerate(account_types):
    data = curve_data[acc_type]
    ax1.plot(data['recall'], data['precision'], color=colors[i], lw=1.5, label=f"{acc_type}")

ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('(a) Precision-Recall Curves', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.legend(loc='lower left', frameon=True, framealpha=0.9)

# Plot (b): ROC curves
ax2 = plt.subplot(gs[0, 1])
for i, acc_type in enumerate(account_types):
    data = curve_data[acc_type]
    ax2.plot(data['fpr'], data['tpr'], color=colors[i], lw=1.5, 
             label=f"{acc_type} (AUC = {data['auc']:.3f})")

ax2.plot([0, 1], [0, 1], 'k--', lw=1)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('(b) ROC Curves', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.02])
ax2.legend(loc='lower right', frameon=True, framealpha=0.9)

# Plot (c): F1-score comparison
ax3 = plt.subplot(gs[1, 0])
bar_width = 0.2
x = np.arange(len(account_types))

for i, method in enumerate(methods[:3]):  # Top-3 methods only
    ax3.bar(x + (i-1)*bar_width, f1_scores[method], bar_width, 
            label=method, color=method_colors[i], edgecolor='black', linewidth=0.5)

ax3.set_xlabel('Account Type')
ax3.set_ylabel('F1-Score (%)')
ax3.set_title('(c) F1-Score Comparison', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(account_types, rotation=30, ha='right')
ax3.set_ylim([65, 100])
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
ax3.legend(loc='lower right', frameon=True, framealpha=0.9)

# Plot (d): Confusion Matrix
ax4 = plt.subplot(gs[1, 1])
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=account_types, yticklabels=account_types, 
            cbar_kws={'label': '%'}, annot_kws={"size": 6}, ax=ax4)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('True')
ax4.set_title('(d) Confusion Matrix', fontweight='bold')
plt.setp(ax4.get_xticklabels(), rotation=30, ha='right')
plt.setp(ax4.get_yticklabels(), rotation=0)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Save figure
plt.savefig('performance_visualization.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('performance_visualization.png', format='png', bbox_inches='tight', dpi=300)
plt.close()

print("Figure created and saved as 'performance_visualization.pdf' and 'performance_visualization.png'")