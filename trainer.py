#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training and evaluation module for ETHAN framework.

This module implements training loops, evaluation metrics, early stopping, 
and model selection criteria.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import AverageMeter, EarlyStopping, save_model

class ModelTrainer:
    """Trainer class for ETHAN model."""
    
    def __init__(self, model, config, device, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        if config.use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
            
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=20,
            min_delta=0.0001,
            mode='max'
        )
        
        # Initialize metrics tracking
        self.train_metrics = {
            'loss': [],
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'auc': []
        }
        
        self.val_metrics = {
            'loss': [],
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'auc': []
        }
        
        # Best validation metrics
        self.best_val_metric = 0.0
        self.best_epoch = 0
        
    def train(self, train_loader, val_loader, epochs):
        """Train the model for the specified number of epochs."""
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader, "Validation")
            
            # Update learning rate scheduler if used
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['f1'])
                
            # Track metrics
            for key in train_metrics:
                self.train_metrics[key].append(train_metrics[key])
                
            for key in val_metrics:
                self.val_metrics[key].append(val_metrics[key])
                
            # Check early stopping
            improved = self.early_stopping(val_metrics['f1'])
            
            if improved:
                self.best_val_metric = val_metrics['f1']
                self.best_epoch = epoch
                
                # Save best model
                save_model(
                    self.model,
                    os.path.join(self.config.output_dir, f"{self.config.experiment_name}_best.pth"),
                    epoch,
                    self.optimizer,
                    val_metrics
                )
                
            # Log progress
            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Check for early stopping
            if self.early_stopping.should_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        self.logger.info(f"Best validation F1: {self.best_val_metric:.4f} at epoch {self.best_epoch}")
        
        return self.best_val_metric, self.best_epoch
        
    def train_epoch(self, data_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        
        # Metrics tracking
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            # Move data to device
            if len(batch) == 5:  # With augmentation
                static_data, static_data_aug, dynamic_data, dynamic_data_aug, labels = batch
                
                # Unpack static data
                x_static, edge_index_static, edge_attr_static, batch_static = [
                    item.to(self.device) if item is not None else None for item in static_data
                ]
                
                # Unpack augmented static data
                x_static_aug, edge_index_static_aug, edge_attr_static_aug, batch_static_aug = [
                    item.to(self.device) if item is not None else None for item in static_data_aug
                ]
                
                # Unpack dynamic data
                x_seq, edge_index_seq, edge_attr_seq, batch_seq = dynamic_data
                x_seq = [item.to(self.device) for item in x_seq]
                edge_index_seq = [item.to(self.device) for item in edge_index_seq]
                edge_attr_seq = [item.to(self.device) if item is not None else None for item in edge_attr_seq]
                batch_seq = [item.to(self.device) if item is not None else None for item in batch_seq]
                
                # Unpack augmented dynamic data
                x_seq_aug, edge_index_seq_aug, edge_attr_seq_aug, batch_seq_aug = dynamic_data_aug
                x_seq_aug = [item.to(self.device) for item in x_seq_aug]
                edge_index_seq_aug = [item.to(self.device) for item in edge_index_seq_aug]
                edge_attr_seq_aug = [item.to(self.device) if item is not None else None for item in edge_attr_seq_aug]
                batch_seq_aug = [item.to(self.device) if item is not None else None for item in batch_seq_aug]
                
                # Move labels to device
                labels = labels.to(self.device)
                
                # Forward pass with augmentation
                output_dict = self.model(
                    (x_static, edge_index_static, edge_attr_static, batch_static),
                    (x_seq, edge_index_seq, edge_attr_seq, batch_seq),
                    return_embeddings=True
                )
                
                output_dict_aug = self.model(
                    (x_static_aug, edge_index_static_aug, edge_attr_static_aug, batch_static_aug),
                    (x_seq_aug, edge_index_seq_aug, edge_attr_seq_aug, batch_seq_aug),
                    return_embeddings=True
                )
                
                # Compute loss with contrastive learning
                loss, loss_dict = self.model.loss_function(
                    output_dict['logits'], 
                    labels,
                    static_logits=output_dict['static_logits'],
                    dynamic_logits=output_dict['dynamic_logits'],
                    static_emb=output_dict['fused_embedding'],
                    dynamic_emb=output_dict_aug['fused_embedding'],
                    kl_div=output_dict.get('static_kl', None),
                    lambda_kl=self.config.lambda_kl,
                    lambda_contrast=self.config.lambda_contrast
                )
                
            else:  # Without augmentation
                static_data, dynamic_data, labels = batch
                
                # Unpack static data
                x_static, edge_index_static, edge_attr_static, batch_static = [
                    item.to(self.device) if item is not None else None for item in static_data
                ]
                
                # Unpack dynamic data
                x_seq, edge_index_seq, edge_attr_seq, batch_seq = dynamic_data
                x_seq = [item.to(self.device) for item in x_seq]
                edge_index_seq = [item.to(self.device) for item in edge_index_seq]
                edge_attr_seq = [item.to(self.device) if item is not None else None for item in edge_attr_seq]
                batch_seq = [item.to(self.device) if item is not None else None for item in batch_seq]
                
                # Move labels to device
                labels = labels.to(self.device)
                
                # Forward pass
                output_dict = self.model(
                    (x_static, edge_index_static, edge_attr_static, batch_static),
                    (x_seq, edge_index_seq, edge_attr_seq, batch_seq),
                    return_embeddings=True
                )
                
                # Compute loss without contrastive learning
                loss, loss_dict = self.model.loss_function(
                    output_dict['logits'], 
                    labels,
                    static_logits=output_dict['static_logits'],
                    dynamic_logits=output_dict['dynamic_logits'],
                    kl_div=output_dict.get('static_kl', None),
                    lambda_kl=self.config.lambda_kl,
                    lambda_contrast=0.0
                )
                
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            batch_size = labels.size(0)
            losses.update(loss.item(), batch_size)
            
            # Get predictions
            _, preds = torch.max(output_dict['logits'], dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'batch_loss': f"{loss.item():.4f}"
            })
            
        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        metrics = self.compute_metrics(all_preds, all_labels)
        
        return losses.avg, metrics
        
    def evaluate(self, data_loader, phase="Validation"):
        """Evaluate the model on the provided data loader."""
        self.model.eval()
        
        # Metrics tracking
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        all_probs = []
        
        # No gradient computation
        with torch.no_grad():
            # Progress bar
            pbar = tqdm(data_loader, desc=f"[{phase}]")
            
            for batch in pbar:
                # Unpack batch
                static_data, dynamic_data, labels = batch
                
                # Unpack static data
                x_static, edge_index_static, edge_attr_static, batch_static = [
                    item.to(self.device) if item is not None else None for item in static_data
                ]
                
                # Unpack dynamic data
                x_seq, edge_index_seq, edge_attr_seq, batch_seq = dynamic_data
                x_seq = [item.to(self.device) for item in x_seq]
                edge_index_seq = [item.to(self.device) for item in edge_index_seq]
                edge_attr_seq = [item.to(self.device) if item is not None else None for item in edge_attr_seq]
                batch_seq = [item.to(self.device) if item is not None else None for item in batch_seq]
                
                # Move labels to device
                labels = labels.to(self.device)
                
                # Forward pass
                output_dict = self.model(
                    (x_static, edge_index_static, edge_attr_static, batch_static),
                    (x_seq, edge_index_seq, edge_attr_seq, batch_seq),
                    return_embeddings=True
                )
                
                # Compute loss
                loss, _ = self.model.loss_function(
                    output_dict['logits'], 
                    labels,
                    static_logits=output_dict['static_logits'],
                    dynamic_logits=output_dict['dynamic_logits'],
                    kl_div=output_dict.get('static_kl', None),
                    lambda_kl=0.0,  # No KL loss during evaluation
                    lambda_contrast=0.0  # No contrastive loss during evaluation
                )
                
                # Update metrics
                batch_size = labels.size(0)
                losses.update(loss.item(), batch_size)
                
                # Get predictions and probabilities
                probs = torch.softmax(output_dict['logits'], dim=1)
                _, preds = torch.max(probs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses.avg:.4f}"
                })
                
        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        metrics = self.compute_metrics(all_preds, all_labels, all_probs)
        
        # Log results
        self.logger.info(
            f"{phase} Results - "
            f"Loss: {losses.avg:.4f}, Acc: {metrics['acc']:.4f}, "
            f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, AUC: {metrics['auc']:.4f}"
        )
        
        return losses.avg, metrics
        
    def compute_metrics(self, y_pred, y_true, y_prob=None):
        """Compute classification metrics."""
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC calculation if probabilities are provided (multiclass)
        auc = 0.0
        if y_prob is not None:
            # One-hot encode true labels
            classes = len(y_prob[0])
            y_true_onehot = np.zeros((len(y_true), classes))
            for i, val in enumerate(y_true):
                y_true_onehot[i, val] = 1
                
            # Calculate AUC (one-vs-rest)
            try:
                auc = roc_auc_score(y_true_onehot, y_prob, multi_class='ovr', average='weighted')
            except ValueError:
                # Fallback if AUC calculation fails (e.g., missing class)
                auc = 0.0
                
        metrics = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics
        
    def predict(self, data_loader, return_probs=False, n_samples=1):
        """Generate predictions for the provided data loader."""
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_uncertainties = []
        
        # No gradient computation
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating predictions"):
                # Unpack batch
                static_data, dynamic_data, _ = batch
                
                # Unpack static data
                x_static, edge_index_static, edge_attr_static, batch_static = [
                    item.to(self.device) if item is not None else None for item in static_data
                ]
                
                # Unpack dynamic data
                x_seq, edge_index_seq, edge_attr_seq, batch_seq = dynamic_data
                x_seq = [item.to(self.device) for item in x_seq]
                edge_index_seq = [item.to(self.device) for item in edge_index_seq]
                edge_attr_seq = [item.to(self.device) if item is not None else None for item in edge_attr_seq]
                batch_seq = [item.to(self.device) if item is not None else None for item in batch_seq]
                
                # Multiple forward passes for uncertainty estimation
                if n_samples > 1 and self.config.use_bayesian_layers:
                    # Initialize prediction storage
                    batch_size = batch_static.max().item() + 1 if batch_static is not None else 1
                    class_probs = torch.zeros((batch_size, self.config.num_classes)).to(self.device)
                    
                    # Multiple forward passes
                    for _ in range(n_samples):
                        # Forward pass
                        logits, _, _, epistemic_uncertainty = self.model(
                            (x_static, edge_index_static, edge_attr_static, batch_static),
                            (x_seq, edge_index_seq, edge_attr_seq, batch_seq),
                            n_samples=1
                        )
                        
                        # Accumulate probabilities
                        class_probs += torch.softmax(logits, dim=1)
                        
                    # Average probabilities
                    class_probs /= n_samples
                    
                    # Get predictions
                    _, preds = torch.max(class_probs, dim=1)
                    
                    # Store results
                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(class_probs.cpu().numpy())
                    all_uncertainties.append(epistemic_uncertainty.cpu().numpy() if epistemic_uncertainty is not None else None)
                    
                else:
                    # Single forward pass
                    logits, calibrated_static, calibrated_dynamic, epistemic_uncertainty = self.model(
                        (x_static, edge_index_static, edge_attr_static, batch_static),
                        (x_seq, edge_index_seq, edge_attr_seq, batch_seq)
                    )
                    
                    # Get predictions
                    class_probs = torch.softmax(logits, dim=1)
                    _, preds = torch.max(class_probs, dim=1)
                    
                    # Store results
                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(class_probs.cpu().numpy())
                    all_uncertainties.append(epistemic_uncertainty.cpu().numpy() if epistemic_uncertainty is not None else None)
                    
        # Concatenate results
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        
        # Process uncertainties if available
        if all_uncertainties[0] is not None:
            all_uncertainties = np.concatenate(all_uncertainties)
        else:
            all_uncertainties = None
            
        if return_probs:
            return all_preds, all_probs, all_uncertainties
        else:
            return all_preds
            
    def analyze_errors(self, data_loader, plot_path=None):
        """Analyze prediction errors and generate visualizations."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        # Generate predictions
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Analyzing errors"):
                # Unpack batch
                static_data, dynamic_data, labels = batch
                
                # Unpack static data
                x_static, edge_index_static, edge_attr_static, batch_static = [
                    item.to(self.device) if item is not None else None for item in static_data
                ]
                
                # Unpack dynamic data
                x_seq, edge_index_seq, edge_attr_seq, batch_seq = dynamic_data
                x_seq = [item.to(self.device) for item in x_seq]
                edge_index_seq = [item.to(self.device) for item in edge_index_seq]
                edge_attr_seq = [item.to(self.device) if item is not None else None for item in edge_attr_seq]
                batch_seq = [item.to(self.device) if item is not None else None for item in batch_seq]
                
                # Move labels to device
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _, _, _ = self.model(
                    (x_static, edge_index_static, edge_attr_static, batch_static),
                    (x_seq, edge_index_seq, edge_attr_seq, batch_seq)
                )
                
                # Get predictions
                _, preds = torch.max(logits, dim=1)
                
                # Store results
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        # Concatenate results
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate per-class metrics
        class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        # Generate confusion matrix plot
        if plot_path is not None:
            plt.figure(figsize=(10, 8))
            class_names = ["Exchange", "ICO-Wallet", "Mining", "Phish/Hack", "DeFi", "Bridge"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, "confusion_matrix.png"), dpi=300)
            plt.close()
            
            # Per-class metrics plot
            plt.figure(figsize=(12, 6))
            x = np.arange(len(class_names))
            width = 0.25
            
            plt.bar(x - width, class_precision, width, label='Precision')
            plt.bar(x, class_recall, width, label='Recall')
            plt.bar(x + width, class_f1, width, label='F1')
            
            plt.xlabel('Account Type')
            plt.ylabel('Score')
            plt.title('Per-class Performance Metrics')
            plt.xticks(x, class_names, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, "per_class_metrics.png"), dpi=300)
            plt.close()
            
        # Log error analysis
        self.logger.info("Error Analysis:")
        for i, class_name in enumerate(["Exchange", "ICO-Wallet", "Mining", "Phish/Hack", "DeFi", "Bridge"]):
            self.logger.info(
                f"  {class_name}: Precision={class_precision[i]:.4f}, "
                f"Recall={class_recall[i]:.4f}, F1={class_f1[i]:.4f}"
            )
            
        # Return analysis results
        return {
            'confusion_matrix': cm,
            'class_precision': class_precision,
            'class_recall': class_recall,
            'class_f1': class_f1
        }