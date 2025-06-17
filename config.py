#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration module for ETHAN framework.

This module handles all hyperparameters, architectural choices, and experimental settings.
"""

import os
import yaml
import argparse
import datetime
from typing import Dict, Any, List

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ETHAN: Ethereum Transaction Hierarchical Analysis Network')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='./data', 
                        help='Path to the dataset directory')
    parser.add_argument('--target_label', type=str, default='ico-wallet',
                        help='Target account type for de-anonymization (exchange, ico-wallet, mining, phish-hack, defi, bridge)')
    parser.add_argument('--hop', type=int, default=2, 
                        help='Number of hops for neighborhood sampling')
    parser.add_argument('--max_neighbors', type=int, default=20, 
                        help='Maximum number of neighbors to sample')
    parser.add_argument('--edge_sampling_strategy', type=str, default='averVolume',
                        help='Strategy for edge sampling (averVolume, Times, Volume)')
    parser.add_argument('--num_time_steps', type=int, default=10,
                        help='Number of time steps for dynamic graph construction')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        help='Fusion type for static and dynamic embeddings (cross_attention, concat)')
    parser.add_argument('--use_bayesian_layers', action='store_true',
                        help='Use Bayesian layers for uncertainty quantification')
    parser.add_argument('--use_cross_attention', action='store_true',
                        help='Use cross-attention mechanism for fusion')
                        
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use graph augmentation for training')
    parser.add_argument('--lambda_kl', type=float, default=0.1,
                        help='Weight for KL divergence loss')
    parser.add_argument('--lambda_contrast', type=float, default=0.1,
                        help='Weight for contrastive loss')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler')
                        
    # System parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for logs and models')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--test_only', action='store_true',
                        help='Test only mode')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained model for testing only')
    
    # Parse args
    args = parser.parse_args()
    
    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"ETHAN_{args.target_label}_{timestamp}"
        
    # Load config file if provided
    if args.config_file is not None:
        args = load_config(args)
        
    return args

def load_config(args):
    """Load configuration from file."""
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
        
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    # Update args with config values
    for key, value in config.items():
        setattr(args, key, value)
        
    return args

class Config:
    """Configuration class for ETHAN framework."""
    
    def __init__(self, args):
        """Initialize configuration from arguments."""
        # Data parameters
        self.data_root = args.data_root
        self.target_label = args.target_label
        self.hop = args.hop
        self.max_neighbors = args.max_neighbors
        self.edge_sampling_strategy = args.edge_sampling_strategy
        self.num_time_steps = args.num_time_steps
        
        # Model parameters
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.fusion_type = args.fusion_type
        self.use_bayesian_layers = args.use_bayesian_layers
        self.use_cross_attention = args.use_cross_attention
        
        # Training parameters
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.use_augmentation = args.use_augmentation
        self.lambda_kl = args.lambda_kl
        self.lambda_contrast = args.lambda_contrast
        self.use_scheduler = args.use_scheduler
        
        # System parameters
        self.gpu_id = args.gpu_id
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.output_dir = args.output_dir
        self.experiment_name = args.experiment_name
        self.test_only = args.test_only
        self.pretrained_path = args.pretrained_path
        
        # Computed parameters
        self.num_classes = 6  # Total number of account types
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save(self, path=None):
        """Save configuration to file."""
        if path is None:
            path = os.path.join(self.output_dir, f"{self.experiment_name}_config.yaml")
            
        config_dict = {k: v for k, v in self.__dict__.items()}
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    def __str__(self):
        """String representation of configuration."""
        return str(self.__dict__)
        
    def to_dict(self):
        """Convert to dictionary."""
        return self.__dict__.copy()