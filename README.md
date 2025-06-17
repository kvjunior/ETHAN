# ETHAN: Ethereum Transaction Hierarchical Analysis Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Dual-Perspective GNN for Ethereum De-anonymization with Bayesian Uncertainty"** submitted to IEEE Transactions on Information Forensics and Security.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Reproducing Results](#reproducing-results)
- [Advanced Usage](#advanced-usage)
- [Experimental Results](#experimental-results)

## 🔍 Overview

ETHAN (Ethereum Transaction Hierarchical Analysis Network) is a state-of-the-art framework for de-anonymizing Ethereum accounts through dual-perspective graph neural networks with Bayesian uncertainty quantification. The framework achieves **97.82% average F1-score** across six account categories, surpassing previous methods by over 12%.

### Key Innovations
- **Dual-Perspective Architecture**: Combines static transaction graphs capturing persistent patterns with dynamic temporal graphs modeling behavioral evolution
- **Bayesian Uncertainty Quantification**: Distinguishes between aleatoric (data) and epistemic (model) uncertainty for risk-based decision making
- **Cross-Attention Fusion**: Enables bidirectional information exchange between static and dynamic representations
- **Adaptive Calibration**: Ensemble of six calibration methods reducing Expected Calibration Error by 78%

## ✨ Key Features

- 🎯 **High Accuracy**: 97.82% average F1-score across all account types
- 🔍 **Uncertainty Estimation**: Reliable confidence scores with 0.745 correlation to actual errors
- ⚡ **Efficient Processing**: 203.5 accounts/second on single GPU
- 🔧 **Modular Design**: Easy to extend and customize components
- 📊 **Comprehensive Evaluation**: Extensive metrics and visualization tools
- 🔐 **Production Ready**: Includes deployment utilities and optimization options

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.3+ compatible GPU (recommended: NVIDIA RTX 3090 or better)
- 32GB+ system RAM
- 100GB+ free disk space for datasets

### Step 1: Clone Repository
git clone https://anonymous.4open.science/r/ETHAN-464A.git
cd ETHAN-464A

### Step 2: Create Virtual Environment

python -m venv ethan_env
source ethan_env/bin/activate  # On Windows: ethan_env\Scripts\activate

### Step 3: Install Dependencies

pip install -r requirements.txt

### Step 4: Verify Installation

python scripts/verify_installation.py

## 📊 Dataset Preparation

### Obtaining Ethereum Transaction Data

1. **Download labeled accounts** from Etherscan Label Cloud:

python scripts/download_labels.py --output data/labels/


2. **Fetch transaction data** using Ethereum node or API:

python scripts/fetch_transactions.py \
    --labels data/labels/labeled_accounts.csv \
    --output data/transactions/ \
    --api-key YOUR_ETHERSCAN_API_KEY \
    --start-date 2015-08-01 \
    --end-date 2024-02-28

3. **Preprocess and extract features**:
python data_processor.py \
    --input data/transactions/ \
    --output data/processed/ \
    --num-workers 16

### Dataset Structure
data/
├── labels/
│   ├── exchange_labels.csv
│   ├── ico_wallet_labels.csv
│   ├── mining_labels.csv
│   ├── phishing_hack_labels.csv
│   ├── defi_labels.csv
│   └── bridge_labels.csv
├── transactions/
│   └── [address].csv
├── processed/
│   ├── node_features.csv
│   ├── static_graphs/
│   └── dynamic_graphs/
└── splits/
    └── [label]_splits.npz


## 🚀 Quick Start

### Training ETHAN Model


python ethan.py \
    --data_root ./data \
    --target_label phish-hack \
    --epochs 200 \
    --batch_size 32 \
    --gpu_id 0 \
    --experiment_name ethan_phishing_detection


### Evaluation Only

python ethan.py \
    --data_root ./data \
    --target_label phish-hack \
    --test_only \
    --pretrained_path ./checkpoints/ethan_phishing_best.pth


### Inference on New Accounts


from models import ETHANModel
from utils import load_model
import torch

# Load pretrained model
model = ETHANModel(
    node_features=15,
    edge_features=1,
    hidden_dim=128,
    output_dim=6,
    num_layers=3,
    num_time_steps=10
)
model = load_model(model, 'checkpoints/ethan_best.pth')
model.eval()

# Prepare account data (see data_processor.py for details)
static_data, dynamic_data = prepare_account_data(account_address)

# Inference with uncertainty
with torch.no_grad():
    logits, calibrated_static, calibrated_dynamic, uncertainty = model(
        static_data, dynamic_data, n_samples=30
    )
    
# Get prediction and confidence
prediction = torch.argmax(logits)
confidence = 1 - uncertainty

## 🏗️ Model Architecture

### Static Graph Encoder
- **Enhanced GAT layers** with edge feature integration
- **Bayesian attention weights** for uncertainty propagation
- **Contrastive learning** with adaptive augmentation

### Dynamic Graph Encoder
- **Temporal GNN** with GRU cells for sequence modeling
- **Time-aware attention** for adaptive temporal aggregation
- **Hierarchical pooling** for multi-scale patterns

### Cross-Attention Fusion
- **Bidirectional attention** between static and dynamic representations
- **Adaptive gating** for controlled information flow
- **Layer normalization** and residual connections

### Uncertainty Quantification
- **Bayesian layers** with scale mixture priors
- **Monte Carlo dropout** with 30 samples
- **Decomposed uncertainty** (aleatoric vs. epistemic)

### Calibration System
- **Ensemble of 6 methods**: Temperature, Vector, Matrix, Beta, Histogram, Isotonic
- **Adaptive weighting** based on validation performance
- **Account-type specific** calibration parameters

## 🔬 Reproducing Results

### Full Experimental Pipeline

1. **Prepare all datasets**:

bash scripts/prepare_all_datasets.sh


2. **Run complete experiments**:

python scripts/run_experiments.py \
    --config configs/full_experiment.yaml \
    --num_seeds 5 \
    --parallel

3. **Generate result tables and figures**:

python scripts/generate_results.py \
    --experiment_dir ./experiments/ \
    --output_dir ./results/

### Ablation Studies

# Component ablation
python scripts/ablation_study.py --component all

# Hyperparameter sensitivity
python scripts/hyperparam_analysis.py --param all --range 0.5,2.0


### Cross-Validation

python scripts/cross_validation.py \
    --folds 5 \
    --stratified \
    --save_predictions


## 🔧 Advanced Usage

### Custom Account Types

# Add new account type to label_map in data_processor.py
self.label_map = {
    'exchange': 0,
    'ico-wallet': 1,
    'mining': 2,
    'phish-hack': 3,
    'defi': 4,
    'bridge': 5,
    'your_new_type': 6  # Add here
}


### Model Configuration

# configs/custom_model.yaml
model:
  hidden_dim: 256  # Increase for more capacity
  num_layers: 4    # Deeper architecture
  num_time_steps: 20  # Finer temporal granularity
  fusion_type: 'cross_attention'
  use_bayesian_layers: true
  
training:
  epochs: 300
  lr: 0.0005
  scheduler: 'cosine'
  early_stopping_patience: 30

### Deployment Optimization


# Convert to TorchScript for production
traced_model = torch.jit.trace(model, (static_data, dynamic_data))
torch.jit.save(traced_model, 'ethan_production.pt')

# Quantization for edge deployment
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.GRU}, dtype=torch.qint8
)

### Real-time Monitoring


from deployment import ETHANMonitor

monitor = ETHANMonitor(
    model_path='checkpoints/ethan_best.pth',
    web3_provider='https://mainnet.infura.io/v3/YOUR_KEY',
    alert_threshold=0.95,
    uncertainty_threshold=0.05
)

monitor.start_monitoring(
    callback=lambda alert: send_alert_to_security_team(alert)
)

## 📈 Experimental Results

### Classification Performance

| Account Type | Precision | Recall | F1-Score | AUC |
|-------------|-----------|---------|----------|------|
| Exchange | 99.51% | 99.41% | 99.46% | 99.87% |
| ICO-Wallet | 97.83% | 97.45% | 97.64% | 99.21% |
| Mining | 98.45% | 98.19% | 98.32% | 99.56% |
| Phish/Hack | 98.92% | 98.49% | 98.71% | 99.63% |
| DeFi | 97.15% | 96.51% | 96.83% | 98.92% |
| Bridge | 96.24% | 95.71% | 95.98% | 98.47% |
| **Average** | **98.02%** | **97.63%** | **97.82%** | **99.28%** |

### Uncertainty-Guided Performance

| Uncertainty Threshold | Coverage | F1-Score | Precision |
|----------------------|----------|----------|-----------|
| 0.025 | 61.37% | 99.95% | 99.96% |
| 0.050 | 78.92% | 99.81% | 99.84% |
| 0.075 | 87.43% | 99.52% | 99.57% |
| 0.100 | 93.71% | 99.16% | 99.23% |

### Computational Performance

- **Training**: 5.83 hours on 4× RTX 3090 GPUs
- **Inference**: 12.83ms per account (single GPU)
- **Batch Processing**: 203.5 accounts/second (batch size 32)
- **Memory Usage**: 19.5GB peak GPU memory

## 📝 Citation

If you use ETHAN in your research, please cite:

@article{anonymous2024ethan,
  title={Dual-Perspective GNN for Ethereum De-anonymization with Bayesian Uncertainty},
  author={Anonymous},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  note={Under Review}
}

### Development Setup

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .

# Type checking
mypy .

## 🙏 Acknowledgments

We thank the Ethereum community for providing labeled account data through Etherscan and XBlock.

## ⚠️ Disclaimer

This tool is designed for legitimate security research and regulatory compliance. Users are responsible for ensuring their use complies with applicable laws and regulations. The authors do not condone or support any illegal activities.
