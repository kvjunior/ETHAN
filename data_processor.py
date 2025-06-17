#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing utilities for ETHAN framework.

This module handles Ethereum transaction data, constructs static and dynamic graph 
representations, and implements graph augmentation techniques.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_sparse import coalesce
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, remove_self_loops, degree, dropout_adj, subgraph
from torch_geometric.nn.conv import MessagePassing
import networkx as nx

# Graph Augmentation Class
class GraphAugmentation:
    """Graph augmentation techniques for data enhancement."""
    
    @staticmethod
    def edge_dropping(edge_index, edge_attr=None, p=0.1, force_undirected=False):
        """Randomly drop edges with probability p."""
        return dropout_adj(edge_index, edge_attr, p=p, force_undirected=force_undirected)
    
    @staticmethod
    def node_dropping(x, edge_index, edge_attr=None, p=0.1):
        """Randomly drop nodes with probability p."""
        num_nodes = x.size(0)
        
        # Generate mask for nodes to keep
        keep_mask = torch.empty(num_nodes, dtype=torch.float32, device=x.device).uniform_(0, 1) > p
        
        # Ensure at least two nodes remain
        if keep_mask.sum() < 2:
            keep_mask[torch.randperm(num_nodes)[:2]] = True
            
        # Get induced subgraph
        edge_index_new, edge_attr_new = subgraph(
            keep_mask, edge_index, edge_attr, relabel_nodes=True, num_nodes=num_nodes
        )
        
        # Get new node features
        x_new = x[keep_mask]
        
        return x_new, edge_index_new, edge_attr_new
    
    @staticmethod
    def feature_masking(x, p=0.1):
        """Randomly mask node features with probability p."""
        feature_mask = torch.empty_like(x, dtype=torch.float32).uniform_(0, 1) > p
        return x * feature_mask
    
    @staticmethod
    def edge_attribute_masking(edge_attr, p=0.1):
        """Randomly mask edge attributes with probability p."""
        if edge_attr is None:
            return None
            
        feature_mask = torch.empty_like(edge_attr, dtype=torch.float32).uniform_(0, 1) > p
        return edge_attr * feature_mask
    
    @staticmethod
    def subgraph_sampling(x, edge_index, edge_attr=None, ratio=0.8):
        """Sample a connected subgraph with a ratio of nodes."""
        num_nodes = x.size(0)
        target_nodes = int(num_nodes * ratio)
        
        if target_nodes < 2:
            return x, edge_index, edge_attr
            
        # Convert to networkx graph for sampling
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Add edges (assuming edge_index is COO format)
        edge_list = edge_index.t().tolist()
        G.add_edges_from(edge_list)
        
        # Find largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        if len(largest_cc) <= target_nodes:
            # If largest component is smaller than target, use it entirely
            nodes_to_keep = list(largest_cc)
        else:
            # Sample a connected subgraph using random walk
            start_node = list(largest_cc)[0]
            nodes_to_keep = [start_node]
            current = start_node
            visited = set([start_node])
            
            while len(nodes_to_keep) < target_nodes:
                neighbors = list(G.neighbors(current))
                unvisited = [n for n in neighbors if n not in visited]
                
                if unvisited:
                    next_node = unvisited[np.random.randint(len(unvisited))]
                    nodes_to_keep.append(next_node)
                    visited.add(next_node)
                    current = next_node
                else:
                    # Restart from a random node already in our subgraph
                    current = nodes_to_keep[np.random.randint(len(nodes_to_keep))]
                    
        # Convert node list to boolean mask
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.tensor(nodes_to_keep)] = True
        
        # Get induced subgraph
        edge_index_new, edge_attr_new = subgraph(
            mask, edge_index, edge_attr, relabel_nodes=True, num_nodes=num_nodes
        )
        
        # Get new node features
        x_new = x[mask]
        
        return x_new, edge_index_new, edge_attr_new
    
    @staticmethod
    def adaptive_edge_dropping(edge_index, edge_attr=None, node_importance=None, p=0.1):
        """Drop edges with probability inversely proportional to node importance."""
        if node_importance is None:
            return GraphAugmentation.edge_dropping(edge_index, edge_attr, p)
            
        num_edges = edge_index.size(1)
        
        # Compute edge importance from node importance
        src_importance = node_importance[edge_index[0]]
        dst_importance = node_importance[edge_index[1]]
        edge_importance = (src_importance + dst_importance) / 2
        
        # Normalize to [0, 1]
        edge_importance = (edge_importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min() + 1e-8)
        
        # Compute drop probability inverse to importance
        drop_prob = p * (1 - edge_importance)
        
        # Generate random mask
        mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
        
        # Apply mask
        edge_index_new = edge_index[:, mask]
        edge_attr_new = edge_attr[mask] if edge_attr is not None else None
        
        return edge_index_new, edge_attr_new
        
    @classmethod
    def apply_augmentations(cls, x, edge_index, edge_attr=None, batch=None, methods=None, p=0.1):
        """Apply a series of augmentations to the graph."""
        if methods is None:
            methods = ['edge_dropping']
            
        # Apply each augmentation method
        for method in methods:
            if method == 'edge_dropping':
                edge_index, edge_attr = cls.edge_dropping(edge_index, edge_attr, p=p)
            elif method == 'node_dropping':
                x, edge_index, edge_attr = cls.node_dropping(x, edge_index, edge_attr, p=p)
            elif method == 'feature_masking':
                x = cls.feature_masking(x, p=p)
            elif method == 'edge_attribute_masking':
                edge_attr = cls.edge_attribute_masking(edge_attr, p=p)
            elif method == 'subgraph_sampling':
                x, edge_index, edge_attr = cls.subgraph_sampling(x, edge_index, edge_attr, ratio=1-p)
            elif method == 'adaptive_edge_dropping':
                # Compute node importance as degree centrality
                node_importance = degree(edge_index[0], x.size(0))
                edge_index, edge_attr = cls.adaptive_edge_dropping(edge_index, edge_attr, node_importance, p=p)
                
        return x, edge_index, edge_attr, batch

# Ethereum Transaction Data Handler
class EthereumTransactionProcessor:
    """Process Ethereum transaction data into graph representations."""
    
    def __init__(self, max_nodes=2000, max_transactions=2000):
        self.max_nodes = max_nodes
        self.max_transactions = max_transactions
        
    def preprocess_transactions(self, transactions):
        """Preprocess raw transaction data."""
        # Normalize transaction timestamps
        timestamps = np.array([float(tx[3]) for tx in transactions])
        min_time = np.min(timestamps)
        max_time = np.max(timestamps)
        time_range = max_time - min_time
        
        for tx in transactions:
            tx[3] = (float(tx[3]) - min_time) / time_range if time_range > 0 else 0
            
        # Sort by normalized time
        transactions = sorted(transactions, key=lambda x: x[3])
        
        # Sample transactions if exceeding maximum
        if len(transactions) > self.max_transactions:
            # Preserve temporal distribution by sampling evenly
            indices = np.linspace(0, len(transactions) - 1, self.max_transactions, dtype=int)
            transactions = [transactions[i] for i in indices]
            
        return transactions
        
    def construct_static_graph(self, transactions, features_map):
        """Construct a static graph from transaction data."""
        # Extract unique nodes
        nodes = set()
        for tx in transactions:
            nodes.add(tx[0])  # sender
            nodes.add(tx[1])  # receiver
            
        # Create node mapping
        node_map = {node: i for i, node in enumerate(nodes)}
        
        # Initialize edges and attributes
        edge_index = []
        edge_attr = []
        
        # Process transactions
        for tx in transactions:
            sender, receiver, value, _ = tx
            
            # Add directed edge: sender -> receiver
            edge_index.append([node_map[sender], node_map[receiver]])
            
            # Edge attribute: [transaction_value]
            edge_attr.append([float(value)])
            
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Prepare node features
        x = torch.zeros((len(nodes), 15), dtype=torch.float)
        
        # Fill node features if available
        for i, node in enumerate(nodes):
            if node in features_map:
                x[i] = torch.tensor(features_map[node], dtype=torch.float)
                
        # Remove self-loops and coalesce multiple edges
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, len(nodes), len(nodes))
        
        return x, edge_index, edge_attr, list(node_map.keys())
        
    def construct_dynamic_graph(self, transactions, features_map, num_time_steps=10):
        """Construct a dynamic graph with temporal slices from transaction data."""
        # Preprocess transactions
        transactions = self.preprocess_transactions(transactions)
        
        # Split into time steps
        time_step_txs = [[] for _ in range(num_time_steps)]
        
        for tx in transactions:
            # Determine time step index
            time_step = min(int(tx[3] * num_time_steps), num_time_steps - 1)
            time_step_txs[time_step].append(tx)
            
        # Extract all unique nodes across all time steps
        all_nodes = set()
        for time_step in time_step_txs:
            for tx in time_step:
                all_nodes.add(tx[0])  # sender
                all_nodes.add(tx[1])  # receiver
                
        # Create global node mapping
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        # Initialize tensors for each time step
        x_seq = []
        edge_index_seq = []
        edge_attr_seq = []
        
        # Process each time step
        for time_step in time_step_txs:
            # Initialize edges and attributes for this time step
            edges = []
            attrs = []
            
            # Process transactions in this time step
            for tx in time_step:
                sender, receiver, value, _ = tx
                
                # Add directed edge: sender -> receiver with global node indices
                edges.append([node_map[sender], node_map[receiver]])
                
                # Edge attribute: [transaction_value]
                attrs.append([float(value)])
                
            # Convert to tensors
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
                edge_attr = torch.tensor(attrs, dtype=torch.float)
                
                # Remove self-loops and coalesce multiple edges
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = coalesce(edge_index, edge_attr, len(node_map), len(node_map))
            else:
                # Empty graph for this time step
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)
                
            # Add to sequences
            edge_index_seq.append(edge_index)
            edge_attr_seq.append(edge_attr)
            
        # Prepare node features (consistent across time steps)
        x = torch.zeros((len(node_map), 15), dtype=torch.float)
        
        # Fill node features if available
        for node, idx in node_map.items():
            if node in features_map:
                x[idx] = torch.tensor(features_map[node], dtype=torch.float)
                
        # Copy node features for each time step
        x_seq = [x for _ in range(num_time_steps)]
        
        return x_seq, edge_index_seq, edge_attr_seq, list(node_map.keys())
        
    def extract_node_features(self, node_address, transactions):
        """Extract 15-dimensional node features from transaction data."""
        # Initialize feature arrays
        sent_txs = []
        received_txs = []
        
        # Separate sent and received transactions
        for tx in transactions:
            sender, receiver, value, timestamp = tx
            if sender == node_address:
                sent_txs.append((float(value), float(timestamp)))
            if receiver == node_address:
                received_txs.append((float(value), float(timestamp)))
                
        # Feature 1-4: Sender account features
        num_txs_sent = len(sent_txs)
        send_total_value = sum(value for value, _ in sent_txs) if sent_txs else 0
        send_avg_value = send_total_value / num_txs_sent if num_txs_sent > 0 else 0
        
        # Calculate send time intervals
        send_time_intervals = []
        if num_txs_sent > 1:
            # Sort by timestamp
            sent_txs.sort(key=lambda x: x[1])
            send_time_intervals = [t2 - t1 for (_, t1), (_, t2) in zip(sent_txs[:-1], sent_txs[1:])]
            
        min_send_interval = min(send_time_intervals) if send_time_intervals else 0
        max_send_interval = max(send_time_intervals) if send_time_intervals else 0
        
        # Feature 5-8: Receiver account features
        num_txs_received = len(received_txs)
        receive_total_value = sum(value for value, _ in received_txs) if received_txs else 0
        receive_avg_value = receive_total_value / num_txs_received if num_txs_received > 0 else 0
        
        # Calculate receive time intervals
        receive_time_intervals = []
        if num_txs_received > 1:
            # Sort by timestamp
            received_txs.sort(key=lambda x: x[1])
            receive_time_intervals = [t2 - t1 for (_, t1), (_, t2) in zip(received_txs[:-1], received_txs[1:])]
            
        min_receive_interval = min(receive_time_intervals) if receive_time_intervals else 0
        max_receive_interval = max(receive_time_intervals) if receive_time_intervals else 0
        
        # Feature 9-12: Transaction fee features
        # Note: If gas data is not available, we'll use proxy calculations
        send_eth_tx_fee = 0.0001 * num_txs_sent  # Approximation
        send_avg_eth_tx_fee = send_eth_tx_fee / num_txs_sent if num_txs_sent > 0 else 0
        receive_eth_tx_fee = 0.0001 * num_txs_received  # Approximation
        receive_avg_eth_tx_fee = receive_eth_tx_fee / num_txs_received if num_txs_received > 0 else 0
        
        # Feature 13-15: Contract features (proxy calculation)
        # Infer if it's likely a contract based on transaction patterns
        is_contract = 1 if (num_txs_received > 3 * num_txs_sent and num_txs_received > 10) else 0
        contract_calls = num_txs_received if is_contract else 0
        contract_creation_time = min([ts for _, ts in received_txs]) if received_txs else 0
        
        # Combine all features
        features = [
            num_txs_sent, send_total_value, send_avg_value, min_send_interval, max_send_interval,
            num_txs_received, receive_total_value, receive_avg_value, min_receive_interval, max_receive_interval,
            send_eth_tx_fee, send_avg_eth_tx_fee, receive_eth_tx_fee, receive_avg_eth_tx_fee,
            contract_calls
        ]
        
        return features
        
    def process_ethereum_data(self, node_address, node_label, transactions, features_map=None, num_time_steps=10):
        """Process Ethereum data for a target node."""
        # If features map not provided, compute it from transactions
        if features_map is None:
            features_map = {}
            
            # Extract all unique nodes
            all_nodes = set()
            for tx in transactions:
                all_nodes.add(tx[0])
                all_nodes.add(tx[1])
                
            # Compute features for each node
            for node in all_nodes:
                features_map[node] = self.extract_node_features(node, transactions)
                
        # Construct static graph
        static_data = self.construct_static_graph(transactions, features_map)
        
        # Construct dynamic graph
        dynamic_data = self.construct_dynamic_graph(transactions, features_map, num_time_steps)
        
        return {
            'static_data': static_data,
            'dynamic_data': dynamic_data,
            'node_address': node_address,
            'node_label': node_label
        }

# Ethereum Dataset Class
class EthereumDataset(Dataset):
    """Dataset for Ethereum account de-anonymization."""
    
    def __init__(self, root, label='ico-wallet', split='train', hop=2, max_neighbors=20, 
                 edge_sampling='averVolume', use_augmentation=False, augmentation_methods=None,
                 augmentation_p=0.1, num_time_steps=10):
        self.root = root
        self.label = label
        self.split = split
        self.hop = hop
        self.max_neighbors = max_neighbors
        self.edge_sampling = edge_sampling
        self.use_augmentation = use_augmentation
        self.augmentation_methods = augmentation_methods or ['edge_dropping', 'feature_masking']
        self.augmentation_p = augmentation_p
        self.num_time_steps = num_time_steps
        
        # Map label names to integer classes
        self.label_map = {
            'exchange': 0,
            'ico-wallet': 1,
            'mining': 2,
            'phish-hack': 3,
            'defi': 4,
            'bridge': 5
        }
        
        # Load label file
        self.addresses, self.labels = self._load_labels()
        
        # Load node features
        self.features_map = self._load_node_features()
        
        # Load transactions
        self.transactions = self._load_transactions()
        
        # Transaction processor
        self.processor = EthereumTransactionProcessor()
        
        # Create train/val/test splits if not already set
        self._create_splits()
        
        # Initialize graph cache for efficiency
        self.graph_cache = {}
        
    def __len__(self):
        """Return dataset size based on split."""
        if self.split == 'train':
            return len(self.train_idx)
        elif self.split == 'val':
            return len(self.val_idx)
        else:  # test
            return len(self.test_idx)
            
    def __getitem__(self, idx):
        """Get graph data for index."""
        # Map to original index
        if self.split == 'train':
            orig_idx = self.train_idx[idx]
        elif self.split == 'val':
            orig_idx = self.val_idx[idx]
        else:  # test
            orig_idx = self.test_idx[idx]
            
        # Check if graph is in cache
        if orig_idx in self.graph_cache:
            graph_data = self.graph_cache[orig_idx]
        else:
            # Process graph data
            address = self.addresses[orig_idx]
            label = self.labels[orig_idx]
            txs = self.transactions.get(address, [])
            
            # Process data
            graph_data = self.processor.process_ethereum_data(
                address, label, txs, self.features_map, self.num_time_steps
            )
            
            # Cache result
            self.graph_cache[orig_idx] = graph_data
            
        # Extract components
        x, edge_index, edge_attr, _ = graph_data['static_data']
        x_seq, edge_index_seq, edge_attr_seq, _ = graph_data['dynamic_data']
        y = torch.tensor(graph_data['node_label'], dtype=torch.long)
        
        # Apply augmentations if enabled and in training mode
        if self.use_augmentation and self.split == 'train':
            x_aug, edge_index_aug, edge_attr_aug, _ = GraphAugmentation.apply_augmentations(
                x, edge_index, edge_attr, None, self.augmentation_methods, self.augmentation_p
            )
            
            # Apply augmentations to each time step in dynamic graph
            x_seq_aug = []
            edge_index_seq_aug = []
            edge_attr_seq_aug = []
            
            for i in range(len(x_seq)):
                x_t_aug, edge_index_t_aug, edge_attr_t_aug, _ = GraphAugmentation.apply_augmentations(
                    x_seq[i], edge_index_seq[i], edge_attr_seq[i], None, 
                    self.augmentation_methods, self.augmentation_p
                )
                x_seq_aug.append(x_t_aug)
                edge_index_seq_aug.append(edge_index_t_aug)
                edge_attr_seq_aug.append(edge_attr_t_aug)
                
            # Return both original and augmented graphs for contrastive learning
            return (
                # Original static graph
                (x, edge_index, edge_attr, None),
                # Augmented static graph
                (x_aug, edge_index_aug, edge_attr_aug, None),
                # Original dynamic graph
                (x_seq, edge_index_seq, edge_attr_seq, None),
                # Augmented dynamic graph
                (x_seq_aug, edge_index_seq_aug, edge_attr_seq_aug, None),
                y
            )
        else:
            # Return only original graph data
            return (
                # Static graph
                (x, edge_index, edge_attr, None),
                # Dynamic graph
                (x_seq, edge_index_seq, edge_attr_seq, None),
                y
            )
    
    def _load_labels(self):
        """Load account addresses and labels."""
        label_path = os.path.join(self.root, f'{self.label}_labels.csv')
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
            
        df = pd.read_csv(label_path)
        addresses = df['address'].tolist()
        labels = [self.label_map.get(row['label'], 0) for _, row in df.iterrows()]
        
        return addresses, labels
        
    def _load_node_features(self):
        """Load precomputed node features."""
        feature_path = os.path.join(self.root, 'node_features.csv')
        
        features_map = {}
        if os.path.exists(feature_path):
            df = pd.read_csv(feature_path)
            for _, row in df.iterrows():
                address = row['address']
                features = [float(row[f'feature_{i}']) for i in range(15)]
                features_map[address] = features
                
        return features_map
        
    def _load_transactions(self):
        """Load transaction data for all addresses."""
        transaction_dir = os.path.join(self.root, 'transactions')
        
        if not os.path.exists(transaction_dir):
            raise FileNotFoundError(f"Transaction directory not found: {transaction_dir}")
            
        # Map of address -> list of transactions
        transactions = {}
        
        # Process each address
        for address in self.addresses:
            tx_path = os.path.join(transaction_dir, f'{address}.csv')
            
            if os.path.exists(tx_path):
                # Load transactions
                df = pd.read_csv(tx_path)
                txs = []
                
                for _, row in df.iterrows():
                    # [sender, receiver, value, timestamp]
                    tx = [row['from'], row['to'], row['value'], row['timestamp']]
                    txs.append(tx)
                    
                transactions[address] = txs
                
        return transactions
        
    def _create_splits(self):
        """Create or load train/val/test splits."""
        split_dir = os.path.join(self.root, 'splits')
        os.makedirs(split_dir, exist_ok=True)
        
        split_file = os.path.join(split_dir, f'{self.label}_splits.npz')
        
        if os.path.exists(split_file):
            # Load existing splits
            splits = np.load(split_file)
            self.train_idx = splits['train']
            self.val_idx = splits['val']
            self.test_idx = splits['test']
        else:
            # Create new splits
            indices = np.arange(len(self.addresses))
            np.random.shuffle(indices)
            
            # 60/20/20 split
            train_ratio, val_ratio = 0.6, 0.2
            
            train_size = int(len(indices) * train_ratio)
            val_size = int(len(indices) * val_ratio)
            
            self.train_idx = indices[:train_size]
            self.val_idx = indices[train_size:train_size + val_size]
            self.test_idx = indices[train_size + val_size:]
            
            # Save splits
            np.savez(split_file, train=self.train_idx, val=self.val_idx, test=self.test_idx)
    
    @property
    def num_classes(self):
        """Return number of classes."""
        return len(set(self.labels))
        
    @property
    def num_node_features(self):
        """Return number of node features."""
        return 15
        
    @property
    def num_edge_features(self):
        """Return number of edge features."""
        return 1

# Static and Dynamic Graph Batching
def collate_static_dynamic(batch):
    """Custom collate function for batching static and dynamic graphs."""
    if len(batch[0]) == 5:  # With augmentation (training)
        # Unpack batch
        static_graphs = [item[0] for item in batch]
        static_graphs_aug = [item[1] for item in batch]
        dynamic_graphs = [item[2] for item in batch]
        dynamic_graphs_aug = [item[3] for item in batch]
        labels = [item[4] for item in batch]
        
        # Batch static graphs
        x_static = torch.cat([g[0] for g in static_graphs], dim=0)
        edge_index_static, edge_attr_static, batch_static = batch_static_graph(static_graphs)
        
        # Batch augmented static graphs
        x_static_aug = torch.cat([g[0] for g in static_graphs_aug], dim=0)
        edge_index_static_aug, edge_attr_static_aug, batch_static_aug = batch_static_graph(static_graphs_aug)
        
        # Batch dynamic graphs (sequence of graphs)
        x_seq, edge_index_seq, edge_attr_seq, batch_seq = batch_dynamic_graph(dynamic_graphs)
        
        # Batch augmented dynamic graphs
        x_seq_aug, edge_index_seq_aug, edge_attr_seq_aug, batch_seq_aug = batch_dynamic_graph(dynamic_graphs_aug)
        
        # Stack labels
        y = torch.stack(labels)
        
        return (
            # Original static graph
            (x_static, edge_index_static, edge_attr_static, batch_static),
            # Augmented static graph
            (x_static_aug, edge_index_static_aug, edge_attr_static_aug, batch_static_aug),
            # Original dynamic graph
            (x_seq, edge_index_seq, edge_attr_seq, batch_seq),
            # Augmented dynamic graph
            (x_seq_aug, edge_index_seq_aug, edge_attr_seq_aug, batch_seq_aug),
            y
        )
    else:  # Without augmentation (validation/testing)
        # Unpack batch
        static_graphs = [item[0] for item in batch]
        dynamic_graphs = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        # Batch static graphs
        x_static = torch.cat([g[0] for g in static_graphs], dim=0)
        edge_index_static, edge_attr_static, batch_static = batch_static_graph(static_graphs)
        
        # Batch dynamic graphs
        x_seq, edge_index_seq, edge_attr_seq, batch_seq = batch_dynamic_graph(dynamic_graphs)
        
        # Stack labels
        y = torch.stack(labels)
        
        return (
            # Static graph
            (x_static, edge_index_static, edge_attr_static, batch_static),
            # Dynamic graph
            (x_seq, edge_index_seq, edge_attr_seq, batch_seq),
            y
        )
        
def batch_static_graph(graphs):
    """Batch a list of static graphs."""
    # Extract components
    xs, edge_indices, edge_attrs, _ = zip(*graphs)
    
    # Cumulative number of nodes
    cum_nodes = 0
    offset_edge_indices = []
    
    # Apply offset to edge indices
    for edge_index in edge_indices:
        offset_edge_indices.append(edge_index + cum_nodes)
        cum_nodes += edge_index.max() + 1
        
    # Concatenate edge indices and attributes
    edge_index = torch.cat(offset_edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)
    
    # Create batch assignment
    batch = []
    for i, x in enumerate(xs):
        batch.extend([i] * x.size(0))
        
    batch = torch.tensor(batch, dtype=torch.long)
    
    return edge_index, edge_attr, batch
    
def batch_dynamic_graph(dynamic_graphs):
    """Batch a list of dynamic graphs (sequences of graphs)."""
    # Extract components
    x_seqs, edge_index_seqs, edge_attr_seqs, _ = zip(*dynamic_graphs)
    
    # Number of time steps
    num_time_steps = len(x_seqs[0])
    
    # Initialize output sequences
    x_seq_batched = []
    edge_index_seq_batched = []
    edge_attr_seq_batched = []
    batch_seq_batched = []
    
    # Process each time step
    for t in range(num_time_steps):
        # Extract graphs at time step t
        xs = [x_seq[t] for x_seq in x_seqs]
        edge_indices = [edge_index_seq[t] for edge_index_seq in edge_index_seqs]
        edge_attrs = [edge_attr_seq[t] for edge_attr_seq in edge_attr_seqs]
        
        # Batch graphs at time step t
        edge_index_t, edge_attr_t, batch_t = batch_static_graph(list(zip(xs, edge_indices, edge_attrs, [None] * len(xs))))
        
        # Concatenate node features
        x_t = torch.cat(xs, dim=0)
        
        # Add to output sequences
        x_seq_batched.append(x_t)
        edge_index_seq_batched.append(edge_index_t)
        edge_attr_seq_batched.append(edge_attr_t)
        batch_seq_batched.append(batch_t)
        
    return x_seq_batched, edge_index_seq_batched, edge_attr_seq_batched, batch_seq_batched