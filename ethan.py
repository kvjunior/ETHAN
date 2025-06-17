#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETHAN: Ethereum Transaction Hierarchical Analysis Network
An advanced framework for de-anonymizing Ethereum accounts by leveraging
dual-perspective graph neural networks with Bayesian uncertainty quantification.

Author: [Your Name]
Affiliation: [Your Affiliation]
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import parse_args, Config
from data_processor import EthereumDataset, GraphAugmentation
from models import ETHANModel
from trainer import ModelTrainer
from utils import setup_logger, set_seed, visualize_results, save_model, load_model

def main():
    """Main execution function for ETHAN framework."""
    # Parse command line arguments
    args = parse_args()
    cfg = Config(args)
    
    # Set up experiment
    set_seed(cfg.seed)
    logger = setup_logger(cfg.output_dir, cfg.experiment_name)
    logger.info(f"Starting ETHAN experiment: {cfg.experiment_name}")
    logger.info(f"Configuration: {cfg}")
    
    # Initialize CUDA devices
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    # Dataset labels: 'exchange', 'ico-wallet', 'mining', 'phish-hack', 'defi', 'bridge'
    train_dataset = EthereumDataset(
        root=cfg.data_root,
        label=cfg.target_label,
        split='train',
        hop=cfg.hop,
        max_neighbors=cfg.max_neighbors,
        edge_sampling=cfg.edge_sampling_strategy,
        augmentation=cfg.use_augmentation
    )
    
    val_dataset = EthereumDataset(
        root=cfg.data_root,
        label=cfg.target_label,
        split='val',
        hop=cfg.hop,
        max_neighbors=cfg.max_neighbors,
        edge_sampling=cfg.edge_sampling_strategy,
        augmentation=False
    )
    
    test_dataset = EthereumDataset(
        root=cfg.data_root,
        label=cfg.target_label,
        split='test',
        hop=cfg.hop,
        max_neighbors=cfg.max_neighbors,
        edge_sampling=cfg.edge_sampling_strategy,
        augmentation=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize model
    logger.info("Initializing ETHAN model...")
    model = ETHANModel(
        node_features=train_dataset.num_node_features,
        edge_features=train_dataset.num_edge_features,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.num_classes,
        num_layers=cfg.num_layers,
        num_time_steps=cfg.num_time_steps,
        fusion_type=cfg.fusion_type,
        bayesian_layers=cfg.use_bayesian_layers,
        cross_attention=cfg.use_cross_attention,
        dropout=cfg.dropout
    ).to(device)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        config=cfg,
        device=device,
        logger=logger
    )
    
    # Train model
    if not args.test_only:
        logger.info("Training ETHAN model...")
        best_val_metric, best_epoch = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs
        )
        logger.info(f"Training completed. Best validation performance: {best_val_metric:.4f} at epoch {best_epoch}")
        
        # Load best model for testing
        model = load_model(model, os.path.join(cfg.output_dir, f"{cfg.experiment_name}_best.pth"))
    else:
        # Load pre-trained model
        model = load_model(model, args.pretrained_path)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Visualize results
    logger.info("Generating visualizations...")
    visualize_results(
        trainer.train_metrics,
        trainer.val_metrics,
        test_metrics,
        cfg.output_dir,
        cfg.experiment_name
    )
    
    logger.info("ETHAN experiment completed successfully.")

if __name__ == "__main__":
    main()