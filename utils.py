#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for ETHAN framework.

This module provides various helper functions, visualization utilities,
and performance metrics tracking.
"""

import os
import time
import random
import logging
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE

# Seed Setting
def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Logging Setup
def setup_logger(log_dir, name):
    """Set up logging configuration."""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    
    # Set log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Model Save/Load
def save_model(model, path, epoch=None, optimizer=None, metrics=None):
    """Save model and related training information."""
    state_dict = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
        
    if metrics is not None:
        state_dict['metrics'] = metrics
        
    torch.save(state_dict, path)
    
def load_model(model, path):
    """Load model from checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
        
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    
    return model

# Metrics Tracking
class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Early Stopping
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=20, min_delta=0.0, mode='max'):
        """
        Initialize early stopping.
        
        Args:
            patience (int): How many epochs to wait before stopping when no improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            mode (str): One of 'min' or 'max' for whether we want to minimize or maximize the metric.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, metric):
        """
        Check if we should stop early based on the metric.
        
        Returns:
            bool: True if metric improved, False otherwise.
        """
        if self.should_stop:
            return False
            
        if self.mode == 'min':
            improved = metric < self.best_metric - self.min_delta
        else:
            improved = metric > self.best_metric + self.min_delta
            
        if improved:
            self.best_metric = metric
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

# Visualization Functions
def visualize_results(train_metrics, val_metrics, test_metrics, output_dir, name):
    """Visualize and save training results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Learning curve
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(train_metrics['loss'], label='Train')
    plt.plot(val_metrics['loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_metrics['acc'], label='Train')
    plt.plot(val_metrics['acc'], label='Validation')
    plt.axhline(y=test_metrics['acc'], color='r', linestyle='--', label=f'Test: {test_metrics["acc"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(train_metrics['f1'], label='Train')
    plt.plot(val_metrics['f1'], label='Validation')
    plt.axhline(y=test_metrics['f1'], color='r', linestyle='--', label=f'Test: {test_metrics["f1"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    
    # Plot AUC
    plt.subplot(2, 2, 4)
    plt.plot(train_metrics['auc'], label='Train')
    plt.plot(val_metrics['auc'], label='Validation')
    plt.axhline(y=test_metrics['auc'], color='r', linestyle='--', label=f'Test: {test_metrics["auc"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_learning_curves.png"), dpi=300)
    plt.close()
    
def visualize_embeddings(model, data_loader, device, output_dir, name):
    """Visualize embeddings using t-SNE."""
    # Extract embeddings
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Unpack batch
            static_data, dynamic_data, labels = batch
            
            # Move data to device
            static_data = [item.to(device) if item is not None else None for item in static_data]
            dynamic_data = [
                [item.to(device) if item is not None else None for item in seq]
                if isinstance(seq, list) else seq
                for seq in dynamic_data
            ]
            
            # Forward pass
            output_dict = model(static_data, dynamic_data, return_embeddings=True)
            
            # Store embeddings and labels
            all_embeddings.append(output_dict['fused_embedding'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    # Concatenate results
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Class names
    class_names = ["Exchange", "ICO-Wallet", "Mining", "Phish/Hack", "DeFi", "Bridge"]
    
    # Create scatter plot for each class
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if np.any(mask):
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=class_name,
                alpha=0.7
            )
    
    plt.legend()
    plt.title("t-SNE Visualization of Account Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_embeddings.png"), dpi=300)
    plt.close()

# Calibration Analysis
def calculate_ece(probabilities, labels, n_bins=15):
    """Calculate Expected Calibration Error."""
    pred_labels = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    accuracies = (pred_labels == labels)
    
    # Bin data
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find examples in bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_size = np.sum(in_bin) / len(labels)
            
            # Add weighted absolute difference
            ece += bin_size * np.abs(bin_acc - bin_conf)
            
    return ece

def plot_calibration_curve(probabilities, labels, output_dir, name, n_bins=15):
    """Plot reliability diagram (calibration curve)."""
    pred_labels = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    accuracies = (pred_labels == labels)
    
    # Bin data
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find examples in bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_size = np.sum(in_bin)
            
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_sizes.append(bin_size)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_sizes.append(0)
            
    # Calculate ECE
    ece = calculate_ece(probabilities, labels, n_bins)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot calibration curve
    plt.bar(bin_centers, bin_accs, width=1.0/n_bins, alpha=0.3, edgecolor='black', label='Accuracy')
    plt.bar(bin_centers, bin_confs, width=1.0/n_bins, alpha=0.3, edgecolor='black', label='Confidence')
    
    # Plot gap
    for i in range(len(bin_centers)):
        plt.plot([bin_centers[i], bin_centers[i]], [bin_accs[i], bin_confs[i]], 'r-', alpha=0.5)
        
    plt.title(f'Calibration Curve (ECE: {ece:.4f})')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_calibration.png"), dpi=300)
    plt.close()
    
    return ece

# Performance Analysis
def analyze_uncertainty(predictions, uncertainties, labels, output_dir, name):
    """Analyze relationship between uncertainty and prediction errors."""
    # Determine correct and incorrect predictions
    correct = predictions == labels
    
    # Plot uncertainty distribution
    plt.figure(figsize=(10, 6))
    
    plt.hist(uncertainties[correct], bins=20, alpha=0.5, label='Correct Predictions')
    plt.hist(uncertainties[~correct], bins=20, alpha=0.5, label='Incorrect Predictions')
    
    plt.xlabel('Epistemic Uncertainty')
    plt.ylabel('Count')
    plt.title('Uncertainty Distribution for Correct and Incorrect Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_uncertainty_distribution.png"), dpi=300)
    plt.close()
    
    # Calculate performance metrics at different uncertainty thresholds
    thresholds = np.linspace(np.min(uncertainties), np.max(uncertainties), 20)
    coverages = []
    accuracies = []
    
    for threshold in thresholds:
        # Mask predictions with uncertainty below threshold
        mask = uncertainties <= threshold
        coverage = np.mean(mask)
        
        if np.sum(mask) > 0:
            accuracy = np.mean(correct[mask])
        else:
            accuracy = 0
            
        coverages.append(coverage)
        accuracies.append(accuracy)
        
    # Plot accuracy vs. coverage (rejection curve)
    plt.figure(figsize=(10, 6))
    
    plt.plot(coverages, accuracies, 'b-', marker='o')
    
    plt.xlabel('Coverage (fraction of data)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Coverage (Rejection Curve)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_rejection_curve.png"), dpi=300)
    plt.close()
    
    return {
        'thresholds': thresholds,
        'coverages': coverages,
        'accuracies': accuracies
    }

# Performance over Time Analysis
def analyze_time_effects(model, time_step_data_loaders, device, output_dir, name):
    """Analyze model performance across different time periods."""
    # Initialize metrics storage
    time_metrics = []
    
    # Evaluate model on each time period
    mod