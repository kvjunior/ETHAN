#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural network architectures for ETHAN framework.

This module implements the static and dynamic graph encoders, 
cross-attention mechanisms, Bayesian uncertainty layers, and fusion modules.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv, GCNConv, GINConv, MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax

# Bayesian Neural Network Components
class BayesianLinear(nn.Module):
    """Bayesian Linear layer with weight and bias uncertainty."""
    
    def __init__(self, in_features, out_features, prior_sigma_1=1.0, prior_sigma_2=0.1, 
                 prior_pi=0.5, posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior distribution parameters
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Posterior parameters initialization
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        
        # Weight parameters
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1))
        
        # Bias parameters
        self.bias_mu = Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
        self.bias_rho = Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1))
        
        # Initialize log_prior and log_variational_posterior
        self.log_prior = 0
        self.log_variational_posterior = 0
        
    def forward(self, x, return_kl=False):
        """Forward pass with KL divergence calculation."""
        # Sample weights and biases from posterior
        weight_sigma = torch.log(1 + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1 + torch.exp(self.bias_rho))
        
        # Sample from normal distribution
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)
        
        # Reparameterization trick
        weight = self.weight_mu + weight_epsilon * weight_sigma
        bias = self.bias_mu + bias_epsilon * bias_sigma
        
        # Compute output
        output = F.linear(x, weight, bias)
        
        if return_kl:
            # Compute KL divergence between posterior and prior
            kl_loss = self._compute_kl_divergence(weight, weight_sigma, bias, bias_sigma)
            return output, kl_loss
        
        return output
    
    def _compute_kl_divergence(self, weight, weight_sigma, bias, bias_sigma):
        """Compute KL divergence between posterior and scale mixture prior."""
        # KL for weights
        weight_log_posterior = self._log_normal(weight, self.weight_mu, weight_sigma.pow(2))
        weight_log_prior = self._log_scale_mixture_prior(weight)
        
        # KL for biases
        bias_log_posterior = self._log_normal(bias, self.bias_mu, bias_sigma.pow(2))
        bias_log_prior = self._log_scale_mixture_prior(bias)
        
        # Total KL divergence
        kl_weight = weight_log_posterior - weight_log_prior
        kl_bias = bias_log_posterior - bias_log_prior
        
        return kl_weight.sum() + kl_bias.sum()
    
    def _log_normal(self, x, mu, sigma_squared):
        """Compute log probability under normal distribution."""
        return -0.5 * (torch.log(2 * math.pi * sigma_squared) + (x - mu).pow(2) / sigma_squared)
    
    def _log_scale_mixture_prior(self, x):
        """Compute log probability under scale mixture prior."""
        log_prior_1 = self._log_normal(x, 0, self.prior_sigma_1**2)
        log_prior_2 = self._log_normal(x, 0, self.prior_sigma_2**2)
        return torch.log(self.prior_pi * torch.exp(log_prior_1) + (1 - self.prior_pi) * torch.exp(log_prior_2))

# Enhanced Graph Attention Layer
class EnhancedGATLayer(MessagePassing):
    """Enhanced Graph Attention layer with edge features and structural integration."""
    
    def __init__(self, in_channels, out_channels, edge_dim=None, heads=8, dropout=0.0,
                 concat=True, negative_slope=0.2, bias=True, use_structural=True):
        super(EnhancedGATLayer, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.use_structural = use_structural
        
        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_v = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None
            
        if use_structural:
            self.lin_struct = nn.Linear(in_channels, heads * out_channels, bias=False)
            
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.att_q = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_k = Parameter(torch.Tensor(1, heads, out_channels))
        
        if edge_dim is not None:
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            
        if self.use_structural:
            nn.init.xavier_uniform_(self.lin_struct.weight)
            
        nn.init.xavier_uniform_(self.att_q)
        nn.init.xavier_uniform_(self.att_k)
        
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.att_edge)
            
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x, edge_index, edge_attr=None, structural_features=None):
        """Forward pass of enhanced GAT layer."""
        H, C = self.heads, self.out_channels
        
        # Transform node features
        q = self.lin_q(x).view(-1, H, C)
        k = self.lin_k(x).view(-1, H, C)
        v = self.lin_v(x).view(-1, H, C)
        
        # Transform structural features if available
        if self.use_structural and structural_features is not None:
            struct_v = self.lin_struct(structural_features).view(-1, H, C)
            v = v + struct_v
            
        # Propagate message
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr, size=None)
        
        # Concatenate or average attention heads
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
            
        # Add bias if specified
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def message(self, q_i, k_j, v_j, edge_attr, index, ptr, size_i):
        """Message function for attention mechanism."""
        # Compute attention scores
        alpha = (q_i * self.att_q) * (k_j * self.att_k)
        
        # Incorporate edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (edge_attr * self.att_edge)
            
        # Sum dimensions and apply activation
        alpha = alpha.sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention scores
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Apply dropout to attention scores
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention weights to value features
        return v_j * alpha.unsqueeze(-1)

# Temporal Graph Neural Network
class TemporalGNN(nn.Module):
    """Temporal Graph Neural Network for processing dynamic graphs."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 num_time_steps=10, use_attention=True, edge_dim=None, dropout=0.0):
        super(TemporalGNN, self).__init__()
        
        self.num_layers = num_layers
        self.num_time_steps = num_time_steps
        self.use_attention = use_attention
        
        # Spatial Graph Neural Networks
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(EnhancedGATLayer(
                in_channels if _ == 0 else hidden_channels,
                hidden_channels,
                edge_dim=edge_dim,
                heads=8,
                dropout=dropout,
                concat=True
            ))
            
        # Temporal recurrent units
        self.gru_cells = nn.ModuleList()
        for _ in range(num_layers):
            self.gru_cells.append(nn.GRUCell(
                hidden_channels * 8,  # 8 attention heads
                hidden_channels * 8
            ))
            
        # Time-step attention weights
        if use_attention:
            self.time_attention = Parameter(torch.Tensor(num_time_steps))
            nn.init.ones_(self.time_attention)
            
        # Output projection
        self.output_proj = nn.Linear(hidden_channels * 8, out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_seq, edge_index_seq, edge_attr_seq=None, batch_seq=None):
        """Process a sequence of graphs through temporal GNN."""
        assert len(x_seq) == self.num_time_steps, "Input sequence length must match num_time_steps"
        
        # Initialize hidden states for each layer
        hidden_states = [None] * self.num_layers
        outputs = []
        
        # Process each time step
        for t in range(self.num_time_steps):
            x = x_seq[t]
            edge_index = edge_index_seq[t]
            edge_attr = edge_attr_seq[t] if edge_attr_seq is not None else None
            batch = batch_seq[t] if batch_seq is not None else None
            
            # Process through GNN layers
            for i in range(self.num_layers):
                if i == 0:
                    # First layer: input is raw node features
                    h = self.gnn_layers[i](x, edge_index, edge_attr)
                else:
                    # Higher layers: input is previous layer's output
                    h = self.gnn_layers[i](h, edge_index, edge_attr)
                    
                # Apply GRU update if we have previous hidden state
                if hidden_states[i] is not None:
                    h = self.gru_cells[i](h, hidden_states[i])
                    
                # Update hidden state
                hidden_states[i] = h
                
                # Apply dropout after each layer
                h = self.dropout(h)
                
            # Store the output of the final layer for this time step
            if batch is not None:
                # Global pooling if batch is provided
                pooled = global_max_pool(h, batch)
            else:
                # Simple mean if no batch info
                pooled = h.mean(dim=0, keepdim=True)
                
            outputs.append(pooled)
            
        # Stack outputs from all time steps
        outputs = torch.stack(outputs, dim=1)  # [batch_size, num_time_steps, hidden_dim]
        
        # Apply time-step attention if specified
        if self.use_attention:
            attn_weights = F.softmax(self.time_attention, dim=0)
            outputs = torch.matmul(outputs, attn_weights.unsqueeze(-1)).squeeze(1)
        else:
            outputs = outputs.mean(dim=1)  # Average across time steps
            
        # Project to output dimension
        return self.output_proj(outputs)

# Static Graph Encoder with Contrastive Learning
class StaticGraphEncoder(nn.Module):
    """Static Graph Encoder with contrastive learning capability."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 edge_dim=None, dropout=0.0, use_bayesian=False):
        super(StaticGraphEncoder, self).__init__()
        
        self.use_bayesian = use_bayesian
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                if use_bayesian:
                    # First layer with Bayesian weights
                    self.conv_layers.append(nn.ModuleDict({
                        'conv': EnhancedGATLayer(
                            in_channels, hidden_channels, edge_dim=edge_dim, 
                            heads=8, dropout=dropout
                        ),
                        'bn': nn.BatchNorm1d(hidden_channels * 8),
                        'bayesian': BayesianLinear(hidden_channels * 8, hidden_channels * 8)
                    }))
                else:
                    self.conv_layers.append(nn.ModuleDict({
                        'conv': EnhancedGATLayer(
                            in_channels, hidden_channels, edge_dim=edge_dim, 
                            heads=8, dropout=dropout
                        ),
                        'bn': nn.BatchNorm1d(hidden_channels * 8)
                    }))
            else:
                if use_bayesian and i == num_layers - 1:
                    # Last layer with Bayesian weights
                    self.conv_layers.append(nn.ModuleDict({
                        'conv': EnhancedGATLayer(
                            hidden_channels * 8, hidden_channels, edge_dim=edge_dim, 
                            heads=8, dropout=dropout
                        ),
                        'bn': nn.BatchNorm1d(hidden_channels * 8),
                        'bayesian': BayesianLinear(hidden_channels * 8, hidden_channels * 8)
                    }))
                else:
                    self.conv_layers.append(nn.ModuleDict({
                        'conv': EnhancedGATLayer(
                            hidden_channels * 8, hidden_channels, edge_dim=edge_dim, 
                            heads=8, dropout=dropout
                        ),
                        'bn': nn.BatchNorm1d(hidden_channels * 8)
                    }))
        
        # Graph readout and projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels * 8, hidden_channels * 4),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels * 2, out_channels)
        
        # Contrastive learning temperature parameter
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, x, edge_index, edge_attr=None, batch=None, return_embeds=False):
        """Forward pass of static graph encoder."""
        # Track KL divergence loss for Bayesian layers
        kl_loss = 0.0
        
        # Graph convolution layers
        for i, layer_dict in enumerate(self.conv_layers):
            if self.use_bayesian and 'bayesian' in layer_dict:
                # Apply graph convolution
                h = layer_dict['conv'](x, edge_index, edge_attr)
                # Apply batch normalization
                h = layer_dict['bn'](h)
                # Apply Bayesian linear with KL tracking
                h, layer_kl = layer_dict['bayesian'](h, return_kl=True)
                kl_loss += layer_kl
            else:
                # Apply graph convolution
                h = layer_dict['conv'](x, edge_index, edge_attr)
                # Apply batch normalization
                h = layer_dict['bn'](h)
                
            # Apply non-linearity except for last layer
            if i < len(self.conv_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.2, training=self.training)
                
            # Update node features
            x = h
            
        # Graph readout
        if batch is not None:
            graph_embedding = global_max_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
            
        # Project embedding for contrastive learning
        z = self.projection(graph_embedding)
        
        # Output projection
        out = self.output_proj(z)
        
        if return_embeds:
            return out, z, kl_loss
        return out
    
    def contrastive_loss(self, z1, z2, batch_size=None):
        """Compute contrastive loss between two sets of embeddings."""
        # Normalize projections
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # Concatenate embeddings from both views
        embeddings = torch.cat([z1_norm, z2_norm], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        mask = 1 - mask
        similarity_matrix = similarity_matrix * mask
        
        # Prepare positive pair indices
        batch_size = batch_size or z1_norm.size(0)
        pos_indices = torch.arange(batch_size, device=similarity_matrix.device)
        pos_indices = torch.cat([pos_indices + batch_size, pos_indices], dim=0)
        
        # Compute log-softmax
        log_prob = F.log_softmax(similarity_matrix, dim=1)
        
        # Compute contrastive loss as negative log likelihood of positive pairs
        loss = -torch.mean(torch.gather(log_prob, 1, pos_indices.unsqueeze(1)))
        
        return loss

# Cross-Attention Module for Static-Dynamic Fusion
class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism to fuse static and dynamic graph representations."""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super(CrossAttentionFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Multi-head cross-attention
        self.static_to_dynamic = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dynamic_to_static = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Layer normalization and residual dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward networks
        self.ff_static = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.ff_dynamic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Output gate for controlling information flow
        self.static_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.dynamic_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, static_embed, dynamic_embed):
        """Forward pass of cross-attention fusion module."""
        batch_size = static_embed.size(0)
        
        # Reshape for attention (seq_len, batch, embed_dim)
        static_q = static_embed.unsqueeze(0)
        dynamic_q = dynamic_embed.unsqueeze(0)
        
        # Static attending to dynamic
        attn_output_s2d, _ = self.static_to_dynamic(
            query=static_q,
            key=dynamic_q,
            value=dynamic_q
        )
        
        # Apply residual connection and layer normalization
        static_enhanced = self.norm1(static_q + self.dropout(attn_output_s2d))
        
        # Feed-forward network for static path
        static_enhanced = self.norm2(static_enhanced + self.dropout(self.ff_static(static_enhanced)))
        
        # Dynamic attending to static
        attn_output_d2s, _ = self.dynamic_to_static(
            query=dynamic_q,
            key=static_q,
            value=static_q
        )
        
        # Apply residual connection and layer normalization
        dynamic_enhanced = self.norm3(dynamic_q + self.dropout(attn_output_d2s))
        
        # Feed-forward network for dynamic path
        dynamic_enhanced = self.norm4(dynamic_enhanced + self.dropout(self.ff_dynamic(dynamic_enhanced)))
        
        # Reshape back to (batch, embed_dim)
        static_enhanced = static_enhanced.squeeze(0)
        dynamic_enhanced = dynamic_enhanced.squeeze(0)
        
        # Compute gates
        static_gate = self.static_gate(torch.cat([static_embed, dynamic_enhanced], dim=1))
        dynamic_gate = self.dynamic_gate(torch.cat([dynamic_embed, static_enhanced], dim=1))
        
        # Apply gates for controlled fusion
        static_final = static_embed * (1 - static_gate) + static_enhanced * static_gate
        dynamic_final = dynamic_embed * (1 - dynamic_gate) + dynamic_enhanced * dynamic_gate
        
        # Concatenate for final fusion
        fused_embedding = torch.cat([static_final, dynamic_final], dim=1)
        
        return fused_embedding

# Probabilistic Calibration Module
class ProbabilisticCalibration(nn.Module):
    """Module for calibrating confidence scores with multiple calibration methods."""
    
    def __init__(self, num_classes, num_methods=6):
        super(ProbabilisticCalibration, self).__init__()
        
        self.num_classes = num_classes
        self.num_methods = num_methods
        
        # Parameters for temperature scaling
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Parameters for vector scaling
        self.vector_scale = nn.Parameter(torch.ones(num_classes))
        self.vector_shift = nn.Parameter(torch.zeros(num_classes))
        
        # Parameters for matrix scaling
        self.matrix_W = nn.Parameter(torch.eye(num_classes))
        self.matrix_b = nn.Parameter(torch.zeros(num_classes))
        
        # Parameters for beta calibration
        self.beta_a = nn.Parameter(torch.ones(num_classes))
        self.beta_b = nn.Parameter(torch.ones(num_classes))
        
        # Adaptive weights for ensemble
        self.method_weights = nn.Parameter(torch.ones(num_methods, 2))  # 2 for static and dynamic
        
    def forward(self, static_logits, dynamic_logits, method='ensemble'):
        """Calibrate prediction confidences."""
        # Convert logits to probabilities
        static_probs = F.softmax(static_logits, dim=1)
        dynamic_probs = F.softmax(dynamic_logits, dim=1)
        
        if method == 'temperature':
            # Temperature scaling
            static_calibrated = self._temperature_scaling(static_probs)
            dynamic_calibrated = self._temperature_scaling(dynamic_probs)
            
        elif method == 'vector':
            # Vector scaling
            static_calibrated = self._vector_scaling(static_probs)
            dynamic_calibrated = self._vector_scaling(dynamic_probs)
            
        elif method == 'matrix':
            # Matrix scaling
            static_calibrated = self._matrix_scaling(static_probs)
            dynamic_calibrated = self._matrix_scaling(dynamic_probs)
            
        elif method == 'beta':
            # Beta calibration
            static_calibrated = self._beta_calibration(static_probs)
            dynamic_calibrated = self._beta_calibration(dynamic_probs)
            
        elif method == 'ensemble':
            # Ensemble of all methods
            static_methods = [
                self._temperature_scaling(static_probs),
                self._vector_scaling(static_probs),
                self._matrix_scaling(static_probs),
                self._beta_calibration(static_probs),
                static_probs,  # Identity (no calibration)
                F.normalize(static_probs, p=1, dim=1)  # Normalize
            ]
            
            dynamic_methods = [
                self._temperature_scaling(dynamic_probs),
                self._vector_scaling(dynamic_probs),
                self._matrix_scaling(dynamic_probs),
                self._beta_calibration(dynamic_probs),
                dynamic_probs,  # Identity (no calibration)
                F.normalize(dynamic_probs, p=1, dim=1)  # Normalize
            ]
            
            # Calculate adaptive weights
            weights = F.softmax(self.method_weights, dim=0)
            
            # Apply weighted ensemble
            static_calibrated = sum(w[0] * m for w, m in zip(weights, static_methods))
            dynamic_calibrated = sum(w[1] * m for w, m in zip(weights, dynamic_methods))
            
        else:
            # Default: no calibration
            static_calibrated = static_probs
            dynamic_calibrated = dynamic_probs
            
        return static_calibrated, dynamic_calibrated
    
    def _temperature_scaling(self, probs):
        """Apply temperature scaling calibration."""
        # Scale logits by temperature
        logits = torch.log(probs + 1e-8)
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=1)
    
    def _vector_scaling(self, probs):
        """Apply vector scaling calibration."""
        logits = torch.log(probs + 1e-8)
        scaled_logits = logits * self.vector_scale + self.vector_shift
        return F.softmax(scaled_logits, dim=1)
    
    def _matrix_scaling(self, probs):
        """Apply matrix scaling calibration."""
        logits = torch.log(probs + 1e-8)
        scaled_logits = torch.matmul(logits, self.matrix_W) + self.matrix_b
        return F.softmax(scaled_logits, dim=1)
    
    def _beta_calibration(self, probs):
        """Apply beta calibration."""
        # Clip probabilities to avoid numerical issues
        probs_clipped = torch.clamp(probs, 1e-8, 1 - 1e-8)
        
        # Apply beta calibration transformation
        calibrated = torch.pow(probs_clipped, self.beta_a) * torch.pow(1 - probs_clipped, self.beta_b)
        
        # Normalize to ensure sum to 1
        return F.normalize(calibrated, p=1, dim=1)
    
    def expected_calibration_error(self, probs, labels, n_bins=10):
        """Compute Expected Calibration Error (ECE)."""
        # Convert to numpy for bin operations
        probs_np = probs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Get predicted classes and their confidences
        pred_classes = np.argmax(probs_np, axis=1)
        confidences = np.max(probs_np, axis=1)
        
        # Accuracy array (1 if correct, 0 if wrong)
        accuracies = (pred_classes == labels_np).astype(np.float32)
        
        # Bin the confidences
        bin_indices = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_sizes = []
        
        for i in range(n_bins):
            bin_mask = (confidences >= bin_indices[i]) & (confidences < bin_indices[i + 1])
            if np.any(bin_mask):
                bin_accs.append(np.mean(accuracies[bin_mask]))
                bin_confs.append(np.mean(confidences[bin_mask]))
                bin_sizes.append(np.sum(bin_mask))
            else:
                bin_accs.append(0)
                bin_confs.append(0)
                bin_sizes.append(0)
                
        # Convert to numpy arrays
        bin_accs = np.array(bin_accs)
        bin_confs = np.array(bin_confs)
        bin_sizes = np.array(bin_sizes)
        
        # Compute ECE
        ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_sizes / np.sum(bin_sizes)))
        
        return torch.tensor(ece, device=probs.device)

# Main ETHAN Model
class ETHANModel(nn.Module):
    """
    ETHAN: Ethereum Transaction Hierarchical Analysis Network
    Main model class integrating all components.
    """
    
    def __init__(self, node_features, edge_features, hidden_dim=128, output_dim=2,
                 num_layers=3, num_time_steps=10, fusion_type='cross_attention',
                 bayesian_layers=True, cross_attention=True, dropout=0.1):
        super(ETHANModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.bayesian_layers = bayesian_layers
        self.cross_attention = cross_attention
        
        # Static Graph Encoder
        self.static_encoder = StaticGraphEncoder(
            in_channels=node_features,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_features,
            dropout=dropout,
            use_bayesian=bayesian_layers
        )
        
        # Dynamic Graph Encoder
        self.dynamic_encoder = TemporalGNN(
            in_channels=node_features,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            num_time_steps=num_time_steps,
            use_attention=True,
            edge_dim=edge_features,
            dropout=dropout
        )
        
        # Fusion module
        if fusion_type == 'cross_attention' and cross_attention:
            self.fusion_module = CrossAttentionFusion(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout
            )
            fusion_output_dim = hidden_dim * 2
        else:
            self.fusion_module = None
            fusion_output_dim = hidden_dim * 2
            
        # Calibration module
        self.calibration = ProbabilisticCalibration(output_dim)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Uncertainty estimation (epistemic uncertainty)
        if bayesian_layers:
            self.uncertainty_estimator = BayesianLinear(
                fusion_output_dim, hidden_dim
            )
        else:
            self.uncertainty_estimator = None
            
    def forward(self, static_data, dynamic_data, return_embeddings=False, n_samples=1):
        """Forward pass of ETHAN model."""
        # Unpack static data
        x_static, edge_index_static, edge_attr_static, batch_static = static_data
        
        # Unpack dynamic data
        x_seq, edge_index_seq, edge_attr_seq, batch_seq = dynamic_data
        
        # Process static graph
        if self.bayesian_layers:
            static_logits, static_emb, static_kl = self.static_encoder(
                x_static, edge_index_static, edge_attr_static, batch_static, return_embeds=True
            )
        else:
            static_logits = self.static_encoder(
                x_static, edge_index_static, edge_attr_static, batch_static
            )
            static_emb = None
            static_kl = 0.0
            
        # Process dynamic graph
        dynamic_logits = self.dynamic_encoder(
            x_seq, edge_index_seq, edge_attr_seq, batch_seq
        )
        
        # Fuse representations
        if self.fusion_type == 'cross_attention' and self.cross_attention:
            fused_emb = self.fusion_module(static_logits, dynamic_logits)
        else:
            fused_emb = torch.cat([static_logits, dynamic_logits], dim=1)
            
        # Calibrate probabilities
        calibrated_static, calibrated_dynamic = self.calibration(static_logits, dynamic_logits)
        
        # Compute epistemic uncertainty if using Bayesian layers
        if self.uncertainty_estimator is not None and n_samples > 1:
            # Multiple forward passes through Bayesian layer
            uncertainty_samples = []
            for _ in range(n_samples):
                output, kl = self.uncertainty_estimator(fused_emb, return_kl=True)
                uncertainty_samples.append(output)
                
            # Compute epistemic uncertainty as variance across samples
            uncertainty_tensor = torch.stack(uncertainty_samples, dim=0)
            epistemic_uncertainty = torch.var(uncertainty_tensor, dim=0).mean(dim=1)
        else:
            epistemic_uncertainty = None
            
        # Final classification
        logits = self.classifier(fused_emb)
        
        if return_embeddings:
            return {
                'logits': logits,
                'static_logits': static_logits,
                'dynamic_logits': dynamic_logits,
                'calibrated_static': calibrated_static,
                'calibrated_dynamic': calibrated_dynamic,
                'fused_embedding': fused_emb,
                'epistemic_uncertainty': epistemic_uncertainty,
                'static_kl': static_kl
            }
        
        return logits, calibrated_static, calibrated_dynamic, epistemic_uncertainty
    
    def loss_function(self, logits, labels, static_logits=None, dynamic_logits=None, 
                      static_emb=None, dynamic_emb=None, kl_div=None, lambda_kl=0.1, 
                      lambda_contrast=0.1):
        """Compute the combined loss function."""
        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)
        
        # Auxiliary classification losses if available
        aux_loss = 0.0
        if static_logits is not None:
            aux_loss += 0.5 * F.cross_entropy(static_logits, labels)
        if dynamic_logits is not None:
            aux_loss += 0.5 * F.cross_entropy(dynamic_logits, labels)
            
        # Contrastive loss if embeddings available
        contrast_loss = 0.0
        if static_emb is not None and dynamic_emb is not None:
            contrast_loss = self.static_encoder.contrastive_loss(static_emb, dynamic_emb)
            
        # KL divergence regularization for Bayesian layers
        kl_loss = 0.0
        if kl_div is not None:
            kl_loss = lambda_kl * kl_div
            
        # Combine all losses
        total_loss = cls_loss + aux_loss + lambda_contrast * contrast_loss + kl_loss
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'aux_loss': aux_loss.item(),
            'contrast_loss': contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else 0.0,
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }