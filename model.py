#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
HiCAPM - Model Module
================================================================================
Manifold spaces:
- Hyperbolic: suitable for hierarchical structures
- Spherical: suitable for cyclic structures
- Euclidean: suitable for flat structures
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

try:
    import geoopt
except ImportError:  # Geoopt is optional at import time; training can still use PyTorch ops.
    geoopt = None
from typing import Dict, List, Tuple, Optional


# ================================================================================
# Manifold space definitions
# ================================================================================

class PoincareBall:
    """
    Poincare ball manifold implementation for hyperbolic space.

    Used to process data with hierarchical structure.
    """
    
    @staticmethod
    def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Apply the exponential map at the origin."""
        v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-8)
        sqrt_c = math.sqrt(c)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    
    @staticmethod
    def logmap0(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        y_norm = torch.clamp(y.norm(dim=-1, keepdim=True), min=1e-8, max=1-1e-5)
        sqrt_c = math.sqrt(c)
        return torch.atanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)
    
    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Perform Mobius addition."""
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = 1 + 2*c*xy + c*c*x2*y2
        
        return num / torch.clamp(denom, min=1e-8)
    
    @staticmethod
    def project(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        max_norm = (1 - eps) / math.sqrt(c)
        norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-8)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)


class SphericalManifold:
    @staticmethod
    def expmap(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the exponential map."""
        v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-8)
        return torch.cos(v_norm) * x + torch.sin(v_norm) * v / v_norm
    
    @staticmethod
    def project(x: torch.Tensor) -> torch.Tensor:
        """Project points onto the unit sphere."""
        return F.normalize(x, p=2, dim=-1)


# ================================================================================
# Curvature calculator
# ================================================================================

class BakryEmeryCurvatureCalculator(nn.Module):
    """
    Bakry-Emery curvature calculator.
    During training, it also adds a learnable BE potential and supports
    first-order curvature evolution driven by task gradients.
    """
    
    def __init__(self, num_entities: int, latent_dim: int, device: str = 'cuda'):
        super().__init__()
        self.num_entities = num_entities
        self.latent_dim = latent_dim
        self.device = device
        
        hidden_dim = max(latent_dim // 2, 1)
        self.curvature_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.be_potential = nn.Parameter(torch.zeros(num_entities))
        self.curvature_eta = nn.Parameter(torch.tensor(0.05))
        self.register_buffer('initial_curvature', torch.zeros(num_entities))
        self.register_buffer('ollivier_ricci_prior', torch.zeros(num_entities))
        self.register_buffer('entropy_prior', torch.zeros(num_entities))
        self.register_buffer('entity_degrees', torch.ones(num_entities))
        
    def _fit_length(self, values: torch.Tensor, target_len: int) -> torch.Tensor:
        if values.shape[0] == target_len:
            return values
        if values.shape[0] > target_len:
            return values[:target_len]
        pad_size = target_len - values.shape[0]
        return torch.cat([values, values[-1:].expand(pad_size)], dim=0)

    def build_topology_priors(self,
                              edge_index: Optional[torch.Tensor],
                              edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.initial_curvature.device
        if edge_index is None or edge_index.numel() == 0:
            curvature = torch.zeros(self.num_entities, device=device)
            self.set_initial_curvature(curvature)
            return curvature

        edge_index = edge_index.to(device)
        src, dst = edge_index
        src = torch.clamp(src, 0, self.num_entities - 1)
        dst = torch.clamp(dst, 0, self.num_entities - 1)

        degrees = torch.zeros(self.num_entities, device=device)
        ones = torch.ones_like(src, dtype=torch.float, device=device)
        degrees.index_add_(0, src, ones)
        degrees.index_add_(0, dst, ones)
        degrees = torch.clamp(degrees, min=1.0)

        edge_degree_gap = torch.abs(torch.log1p(degrees[src]) - torch.log1p(degrees[dst]))
        max_log_degree = torch.clamp(torch.log1p(degrees).max(), min=1.0)
        edge_ricci = 1.0 - edge_degree_gap / max_log_degree
        ricci_sum = torch.zeros(self.num_entities, device=device)
        ricci_count = torch.zeros(self.num_entities, device=device)
        ricci_sum.index_add_(0, src, edge_ricci)
        ricci_sum.index_add_(0, dst, edge_ricci)
        ricci_count.index_add_(0, src, ones)
        ricci_count.index_add_(0, dst, ones)
        ricci_prior = ricci_sum / torch.clamp(ricci_count, min=1.0)

        entropy_prior = torch.log1p(degrees) / max_log_degree
        if edge_type is not None and edge_type.numel() == src.numel():
            edge_type = edge_type.to(device)
            relation_variety = torch.zeros(self.num_entities, device=device)
            relation_variety.index_add_(0, src, torch.ones_like(src, dtype=torch.float, device=device))
            relation_variety.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float, device=device))
            entropy_prior = 0.5 * entropy_prior + 0.5 * torch.log1p(relation_variety) / max_log_degree

        curvature = 0.7 * ricci_prior + 0.3 * entropy_prior
        curvature = 2.0 * curvature - 1.0

        self.ollivier_ricci_prior.copy_(ricci_prior.detach())
        self.entropy_prior.copy_(entropy_prior.detach())
        self.entity_degrees.copy_(degrees.detach())
        self.initial_curvature.copy_(curvature.detach())
        return curvature
        
    def _normalize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal - signal.mean()
        return signal / torch.clamp(signal.abs().mean(), min=1e-6)

    def compute_curvature(self,
                          embeddings: torch.Tensor,
                          task_grad: torch.Tensor = None,
                          curvature_grad: torch.Tensor = None) -> torch.Tensor:
        """Compute curvature using topology priors, a BE potential, and task gradients."""
        n_entities = embeddings.shape[0]
        initial_curvature = self._fit_length(self.initial_curvature, n_entities).to(embeddings.device)
        curvature = initial_curvature

        if curvature_grad is not None:
            curvature_grad = self._fit_length(curvature_grad.detach(), n_entities).to(embeddings.device)
            curvature = initial_curvature + self.curvature_eta * self._normalize_signal(curvature_grad)
        elif task_grad is not None:
            grad_signal = torch.sum(task_grad.detach() * embeddings.detach(), dim=-1)
            curvature = initial_curvature + self.curvature_eta * self._normalize_signal(grad_signal)
        
        return torch.clamp(curvature, -1.0, 1.0)
    
    def set_initial_curvature(self, curvature: torch.Tensor):
        """Set the initial curvature."""
        curvature = self._fit_length(curvature.detach(), self.num_entities).to(self.initial_curvature.device)
        self.initial_curvature.copy_(curvature)
        
    def set_entity_degrees(self, degrees: torch.Tensor):
        """Set entity degrees."""
        degrees = self._fit_length(degrees.detach(), self.num_entities).to(self.entity_degrees.device)
        self.entity_degrees.copy_(degrees)


# ================================================================================
# How1: Hierarchical curvature-weighted manifolds
# ================================================================================

class How1HierarchicalCurvatureWeighting(nn.Module):
    """
    How1: Hierarchical curvature-weighted manifold mechanism.
    
    Dynamically assign weights to three manifolds (hyperbolic, spherical,
    and Euclidean) according to each entity's curvature features:
    """
    
    def __init__(self, num_entities: int, num_relations: int, 
                 latent_dim: int, device: str = 'cuda',
                 warmup_epochs: int = 10):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.latent_dim = latent_dim
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # Entity-level weighting network: g_i(B(kappa_v), R_v)
        self.entity_weight_net = nn.Sequential(
            nn.Linear(2 * latent_dim + 1, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 3),
            nn.Softmax(dim=-1)
        )
        
        # Relation-level weighting network
        self.relation_weight_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 3),
            nn.Softmax(dim=-1)
        )
        
        # Curvature calculator
        self.curvature_calculator = BakryEmeryCurvatureCalculator(
            num_entities, latent_dim, device
        )
        
        self.frozen = True
        
    def _relation_context(self,
                          entity_embeds: torch.Tensor,
                          relation_embeds: torch.Tensor,
                          edge_index: torch.Tensor = None,
                          edge_type: torch.Tensor = None) -> torch.Tensor:
        context = torch.zeros_like(entity_embeds)
        if edge_index is None or edge_type is None or edge_index.numel() == 0:
            return context

        src = edge_index[0]
        valid = (src >= 0) & (src < entity_embeds.shape[0])
        src = src[valid]
        if src.numel() == 0:
            return context

        rel_ids = torch.clamp(edge_type[valid], 0, relation_embeds.shape[0] - 1)
        context.index_add_(0, src, relation_embeds[rel_ids])
        counts = torch.zeros(entity_embeds.shape[0], device=entity_embeds.device)
        counts.index_add_(0, src, torch.ones_like(src, dtype=torch.float, device=entity_embeds.device))
        return context / torch.clamp(counts.unsqueeze(-1), min=1.0)

    def forward(self, entity_embeds: torch.Tensor, 
                relation_embeds: torch.Tensor,
                epoch: int = 0,
                external_curvature: torch.Tensor = None,
                task_grad: torch.Tensor = None,
                curvature_grad: torch.Tensor = None,
                edge_index: torch.Tensor = None,
                edge_type: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass.

        """
        # Compute curvature
        if external_curvature is not None:
            curvature = external_curvature
        else:
            curvature = self.curvature_calculator.compute_curvature(
                entity_embeds,
                task_grad=task_grad,
                curvature_grad=curvature_grad
            )

        if curvature.shape[0] != entity_embeds.shape[0]:
            if curvature.shape[0] > entity_embeds.shape[0]:
                curvature = curvature[:entity_embeds.shape[0]]
            else:
                pad_size = entity_embeds.shape[0] - curvature.shape[0]
                curvature = torch.cat([curvature, curvature[-1:].expand(pad_size)], dim=0)
        
        if epoch < self.warmup_epochs:
            entity_weights = torch.full(
                (entity_embeds.shape[0], 3),
                1.0 / 3.0,
                device=entity_embeds.device,
                dtype=entity_embeds.dtype
            )
            relation_weights = torch.full(
                (relation_embeds.shape[0], 3),
                1.0 / 3.0,
                device=relation_embeds.device,
                dtype=relation_embeds.dtype
            )
            return curvature, entity_weights, relation_weights

        relation_context = self._relation_context(entity_embeds, relation_embeds, edge_index, edge_type)
        entity_input = torch.cat([entity_embeds, relation_context, curvature.unsqueeze(-1)], dim=-1)
        entity_weights = self.entity_weight_net(entity_input)
        
        # Compute relation-level weights
        relation_weights = self.relation_weight_net(relation_embeds)
        
        ramp = min(1.0, max(0.0, (epoch - self.warmup_epochs + 1) / max(float(self.warmup_epochs), 1.0)))
        entity_uniform = torch.full_like(entity_weights, 1.0 / 3.0)
        relation_uniform = torch.full_like(relation_weights, 1.0 / 3.0)
        entity_weights = (1.0 - ramp) * entity_uniform + ramp * entity_weights
        relation_weights = (1.0 - ramp) * relation_uniform + ramp * relation_weights
        return curvature, entity_weights, relation_weights
    
    def freeze_parameters(self):
        """Freeze parameters."""
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_parameters(self):
        """Unfreeze parameters."""
        self.frozen = False
        for param in self.parameters():
            param.requires_grad = True


# ================================================================================
# How2: Curvature-driven asynchronous diffusion
# ================================================================================

class CurvatureDepthMapper(nn.Module):
    """
    Map curvature values to propagation depths.
    """
    
    def __init__(self, min_depth: int = 1, max_depth: int = 6,
                 alpha: float = -1.5, beta: float = 0.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, curvature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map curvature values to propagation depths."""
        curv_min = curvature.min()
        curv_max = curvature.max()
        curvature_norm = (curvature - curv_min) / torch.clamp(curv_max - curv_min, min=1e-8)
        depth_ratio = torch.sigmoid(self.alpha * curvature_norm + self.beta)
        depths = self.min_depth + depth_ratio * (self.max_depth - self.min_depth)
        depths_int = torch.clamp(depths.round().long(), self.min_depth, self.max_depth)
        return depths, depths_int


class How2AsyncDiffusion(nn.Module):
    """
    How2: Curvature-driven asynchronous diffusion mechanism.
    
    Determine information-propagation depth from node curvature features.
    """
    
    def __init__(self, num_entities: int, latent_dim: int, 
                 num_relations: int, device: str = 'cuda'):
        super().__init__()
        self.num_entities = num_entities
        self.latent_dim = latent_dim
        self.device = device
        
        self.depth_mapper = CurvatureDepthMapper(min_depth=1, max_depth=6)
        self.curvature_calculator = BakryEmeryCurvatureCalculator(
            num_entities, latent_dim, device
        )
        
        self.saturation_threshold = 0.9
        
        self.propagation_layers = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(6)
        ])
        self.attention_layers = nn.ModuleList([
            nn.Linear(2 * latent_dim, 1, bias=False) for _ in range(6)
        ])
        
    def _neighbor_attention_aggregate(self,
                                      x: torch.Tensor,
                                      edge_index: torch.Tensor,
                                      layer_idx: int,
                                      edge_type: torch.Tensor = None,
                                      relation_embeds: torch.Tensor = None,
                                      curvature: torch.Tensor = None) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return x

        src, dst = edge_index
        valid = (src >= 0) & (src < x.shape[0]) & (dst >= 0) & (dst < x.shape[0])
        src = src[valid]
        dst = dst[valid]
        if src.numel() == 0:
            return x

        neighbor_message = x[dst]
        if edge_type is not None and relation_embeds is not None and edge_type.numel() == valid.numel():
            edge_type = edge_type[valid]
            edge_type = torch.clamp(edge_type, 0, relation_embeds.shape[0] - 1)
            neighbor_message = neighbor_message + relation_embeds[edge_type]

        edge_feat = torch.cat([x[src], neighbor_message], dim=-1)
        attn_logits = self.attention_layers[layer_idx](edge_feat).squeeze(-1)
        if curvature is not None:
            curvature = curvature.to(x.device)
            curvature_gap = torch.abs(curvature[src] - curvature[dst])
            attn_logits = attn_logits - curvature_gap
        attn_logits = torch.clamp(attn_logits, -10.0, 10.0)
        attn_exp = torch.exp(attn_logits)
        denom = torch.zeros(x.shape[0], device=x.device)
        denom.index_add_(0, src, attn_exp)
        attn = attn_exp / torch.clamp(denom[src], min=1e-8)

        messages = neighbor_message * attn.unsqueeze(-1)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, src, messages)
        return aggregated
        
    def forward(self, entity_embeds: torch.Tensor,
                edge_index: torch.Tensor = None,
                edge_type: torch.Tensor = None,
                relation_embeds: torch.Tensor = None,
                precomputed: Dict = None,
                task_grad: torch.Tensor = None,
                curvature_grad: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        if precomputed is not None and 'curvature' in precomputed:
            curvature = precomputed['curvature']
        else:
            curvature = self.curvature_calculator.compute_curvature(
                entity_embeds,
                task_grad=task_grad,
                curvature_grad=curvature_grad
            )
        
        # Ensure matching dimensions
        if curvature.shape[0] != entity_embeds.shape[0]:
            if curvature.shape[0] > entity_embeds.shape[0]:
                curvature = curvature[:entity_embeds.shape[0]]
            else:
                pad_size = entity_embeds.shape[0] - curvature.shape[0]
                curvature = torch.cat([curvature, curvature[-1:].expand(pad_size)], dim=0)
        
        # Compute propagation depths
        depths, depths_int = self.depth_mapper(curvature)

        output_embeds = entity_embeds.clone()
        
        layer_entity_counts = []
        for d in range(1, len(self.propagation_layers) + 1):
            mask = depths_int >= d
            layer_entity_counts.append(int(mask.sum().item()))
            if mask.sum() > 0:
                layer_idx = min(d - 1, len(self.propagation_layers) - 1)
                aggregated = self._neighbor_attention_aggregate(
                    output_embeds,
                    edge_index,
                    layer_idx,
                    edge_type=edge_type,
                    relation_embeds=relation_embeds,
                    curvature=curvature
                )
                propagated = self.propagation_layers[layer_idx](aggregated)
                propagated = F.relu(propagated)
                
                output_embeds = torch.where(
                    mask.unsqueeze(-1).expand_as(output_embeds),
                    propagated,
                    output_embeds
                )

        saturated_mask = curvature > self.saturation_threshold
        saturation_stats = {
            'saturated_nodes': saturated_mask.sum().item(),
            'total_nodes': curvature.shape[0],
            'saturation_ratio': saturated_mask.float().mean().item(),
            'layer_entity_counts': layer_entity_counts
        }
        
        return output_embeds, depths, depths_int, curvature, saturation_stats


# ================================================================================
# How3: Task alignment weighting
# ================================================================================

class TaskAlignmentModule(nn.Module):
    """
    How3: Task alignment weighting mechanism.

    Generate adaptive weights from Bakry-Emery curvature and task-gradient
    information.
    """
    
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int = 64,
                 use_curvature: bool = True,
                 warmup_epochs: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_curvature = use_curvature
        self.warmup_epochs = warmup_epochs
        
        input_dim = latent_dim + (2 if use_curvature else 0)
        
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.sensitivity = nn.Parameter(torch.tensor(0.1))
        self.current_epoch = 0
        
    def forward(self, entity_emb: torch.Tensor,
                task_grad: torch.Tensor = None,
                bakry_emery_curvature: torch.Tensor = None,
                degree_norm: torch.Tensor = None) -> torch.Tensor:
        n_entities = entity_emb.shape[0]
        device = entity_emb.device

        if self.current_epoch < self.warmup_epochs:
            return torch.full((n_entities, 1), 0.5, device=device, dtype=entity_emb.dtype)
        
        if self.use_curvature:
            if bakry_emery_curvature is not None:
                if bakry_emery_curvature.shape[0] != n_entities:
                    if bakry_emery_curvature.shape[0] > n_entities:
                        bakry_emery_curvature = bakry_emery_curvature[:n_entities]
                    else:
                        pad_size = n_entities - bakry_emery_curvature.shape[0]
                        bakry_emery_curvature = torch.cat([
                            bakry_emery_curvature,
                            bakry_emery_curvature[-1:].expand(pad_size)
                        ], dim=0)
                curvature_feat = torch.sigmoid(bakry_emery_curvature).unsqueeze(1)
            else:
                curvature_feat = torch.zeros((n_entities, 1), device=device)
            
            if degree_norm is not None:
                if degree_norm.shape[0] != n_entities:
                    if degree_norm.shape[0] > n_entities:
                        degree_norm = degree_norm[:n_entities]
                    else:
                        pad_size = n_entities - degree_norm.shape[0]
                        degree_norm = torch.cat([
                            degree_norm,
                            degree_norm[-1:].expand(pad_size)
                        ], dim=0)
                degree_feat = degree_norm.unsqueeze(1) if degree_norm.dim() == 1 else degree_norm
            else:
                degree_feat = torch.zeros((n_entities, 1), device=device)
            
            combined_input = torch.cat([entity_emb, curvature_feat, degree_feat], dim=1)
        else:
            combined_input = entity_emb
        
        base_weight = self.weight_net(combined_input)
        
        if task_grad is not None:
            grad_influence = torch.sum(task_grad * entity_emb, dim=1, keepdim=True)
            grad_influence = grad_influence / torch.clamp(grad_influence.abs().mean(), min=1e-6)
            adjusted_weight = base_weight * (1 + self.sensitivity * grad_influence)
        else:
            adjusted_weight = base_weight
        
        ramp = min(1.0, max(0.0, (self.current_epoch - self.warmup_epochs + 1) / max(float(self.warmup_epochs), 1.0)))
        neutral_weight = torch.full_like(adjusted_weight, 0.5)
        adjusted_weight = (1.0 - ramp) * neutral_weight + ramp * adjusted_weight
        return torch.clamp(adjusted_weight, 0, 1)
    
    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch
        
    def get_training_phase(self) -> Tuple[str, Dict]:
        return 'learning', {'loss_weight': 1.0, 'apply_to_embeds': True}


# ================================================================================
# Graph neural network layers
# ================================================================================

class GCNLayer(nn.Module):
    """Graph convolutional layer."""
    
    def __init__(self, input_dim: int = None, output_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if input_dim is not None and output_dim is not None:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear = None
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        if adj.is_sparse:
            x = torch.sparse.mm(adj, x)
        else:
            x = torch.mm(adj, x)
        
        if self.linear is not None:
            x = self.linear(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x


class RGAT(nn.Module):
    """Relational graph attention network."""
    
    def __init__(self, latent_dim: int, n_hops: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_hops = n_hops
        self.dropout = nn.Dropout(dropout_rate)
        
        self.W = nn.Parameter(torch.empty(2 * latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W)
        
        self.relu = nn.ReLU()
        
    def forward(self, entity_emb: torch.Tensor, 
                relation_emb: torch.Tensor,
                kg: Tuple[torch.Tensor, torch.Tensor],
                mess_dropout: bool = True) -> torch.Tensor:
        """Perform the forward pass."""
        edge_index, edge_type = kg
        entity_res_emb = entity_emb
        
        for _ in range(self.n_hops):
            entity_emb = self._aggregate(entity_emb, relation_emb, edge_index, edge_type)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb, dim=-1)
            entity_res_emb = 0.5 * entity_res_emb + entity_emb
        
        return entity_res_emb
    
    def _aggregate(self, entity_emb: torch.Tensor,
                   relation_emb: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_type: torch.Tensor) -> torch.Tensor:
        """Aggregate information from neighboring nodes."""
        head, tail = edge_index
        n_entities = entity_emb.shape[0]
        
        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]
        
        edge_type_clamped = torch.clamp(edge_type, 0, relation_emb.shape[0] - 1)
        rel_emb = relation_emb[edge_type_clamped]
        
        message = tail_emb + rel_emb
        message = self.relu(message)
        
        agg_emb = torch.zeros_like(entity_emb)
        agg_emb.index_add_(0, head, message)
        
        count = torch.zeros(n_entities, device=entity_emb.device)
        count.index_add_(0, head, torch.ones_like(head, dtype=torch.float))
        count = torch.clamp(count, min=1).unsqueeze(-1)
        agg_emb = agg_emb / count
        
        return agg_emb + entity_emb


# ================================================================================
# Main model
# ================================================================================

class HiCAPM(nn.Module):
    """
    HiCAPM: Hierarchical Curvature-Adaptive Product Manifold model.

    """
    
    def __init__(self, user_num: int, item_num: int, 
                 entity_num: int, relation_num: int,
                 latent_dim: int = 64, n_layers: int = 2,
                 device: str = 'cuda',
                 warmup_epochs: int = 10):
        super().__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.latent_dim = latent_dim
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # Embedding layers
        if geoopt is not None:
            self.user_embeds = geoopt.ManifoldParameter(
                torch.empty(user_num, latent_dim), manifold=geoopt.Euclidean()
            )
            self.entity_embeds = geoopt.ManifoldParameter(
                torch.empty(entity_num, latent_dim), manifold=geoopt.Euclidean()
            )
            self.relation_embeds = geoopt.ManifoldParameter(
                torch.empty(relation_num, latent_dim), manifold=geoopt.Euclidean()
            )
        else:
            self.user_embeds = nn.Parameter(torch.empty(user_num, latent_dim))
            self.entity_embeds = nn.Parameter(torch.empty(entity_num, latent_dim))
            self.relation_embeds = nn.Parameter(torch.empty(relation_num, latent_dim))
        
        nn.init.xavier_uniform_(self.user_embeds)
        nn.init.xavier_uniform_(self.entity_embeds)
        nn.init.xavier_uniform_(self.relation_embeds)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(n_layers)])
        
        # RGAT
        self.rgat = RGAT(latent_dim, n_hops=2)

        self.how1_enhancement = How1HierarchicalCurvatureWeighting(
            entity_num, relation_num, latent_dim, device, warmup_epochs=warmup_epochs
        )

        self.how2_enhancement = How2AsyncDiffusion(
            entity_num, latent_dim, relation_num, device
        )

        self.task_alignment = TaskAlignmentModule(latent_dim, warmup_epochs=warmup_epochs)

        self.unified_curvature = BakryEmeryCurvatureCalculator(
            entity_num, latent_dim, device
        )
        
        # Cached values
        self.how2_stats = None
        self.how3_weights_cache = None
        self.current_entity_weights = None
        self.current_relation_weights = None
        self.current_curvature = None
        self.current_product_entity_embeds = None
        self.current_diffused_entity_embeds = None
        self.current_rec_curvature = None
        self.current_rec_degrees = None
        self._current_epoch = 0
        self._topology_priors_ready = False
        print("[OK] How1 enhanced loss function initialized")
        print("[OK] Unified curvature calculator initialized")
        print("[OK] How1 hierarchical curvature-weighted manifold initialized")
        print("[OK] How2 asynchronous diffusion initialized")
        print("[OK] How3 task alignment initialized")

    def _copy_curvature_state(self,
                              source: BakryEmeryCurvatureCalculator,
                              target: BakryEmeryCurvatureCalculator):
        target.set_initial_curvature(source.initial_curvature)
        target.set_entity_degrees(source.entity_degrees)
        target.ollivier_ricci_prior.copy_(source.ollivier_ricci_prior.detach())
        target.entropy_prior.copy_(source.entropy_prior.detach())

    def initialize_topology_priors(self,
                                   kg: Tuple[torch.Tensor, torch.Tensor] = None,
                                   force: bool = False):
        """Initialize shared curvature priors from the KG topology."""
        if self._topology_priors_ready and not force:
            return

        edge_index = kg[0] if kg is not None else None
        edge_type = kg[1] if kg is not None else None
        self.unified_curvature.build_topology_priors(edge_index, edge_type=edge_type)
        self._copy_curvature_state(self.unified_curvature, self.how1_enhancement.curvature_calculator)
        self._copy_curvature_state(self.unified_curvature, self.how2_enhancement.curvature_calculator)
        self._topology_priors_ready = True

    def _fit_entity_signal(self,
                           signal: torch.Tensor,
                           feature_dim: int = None,
                           fill_value: float = 0.0) -> torch.Tensor:
        if feature_dim is None:
            output = torch.full((self.entity_num,), fill_value, device=self.entity_embeds.device)
        else:
            output = torch.full(
                (self.entity_num, feature_dim),
                fill_value,
                device=self.entity_embeds.device,
                dtype=self.entity_embeds.dtype
            )
        if signal is None:
            return output
        limit = min(signal.shape[0], self.entity_num)
        output[:limit] = signal[:limit].to(output.device)
        return output

    def _rec_task_grad_to_entity_grad(self, task_grad: torch.Tensor) -> torch.Tensor:
        if task_grad is None:
            return None
        entity_grad = torch.zeros_like(self.entity_embeds)
        item_grad = task_grad[self.user_num:self.user_num + self.item_num]
        limit = min(item_grad.shape[0], self.entity_num)
        entity_grad[:limit] = item_grad[:limit].detach()
        return entity_grad

    def _entity_signal_to_rec_nodes(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.to(self.entity_embeds.device)
        rec_signal = torch.zeros(self.user_num + self.item_num, device=signal.device, dtype=signal.dtype)
        limit = min(self.item_num, signal.shape[0])
        rec_signal[self.user_num:self.user_num + limit] = signal[:limit]
        return rec_signal
        
    def forward(self, adj: torch.Tensor,
                kg: Tuple[torch.Tensor, torch.Tensor] = None,
                mess_dropout: bool = True,
                task_grad: torch.Tensor = None,
                curvature_grad: torch.Tensor = None,
                compute_alignment: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass."""
        if kg is not None:
            self.initialize_topology_priors(kg)

        entity_task_grad = self._rec_task_grad_to_entity_grad(task_grad)

        # Encode the knowledge graph
        if kg is not None:
            entity_embeds = self.rgat(self.entity_embeds, self.relation_embeds, kg, mess_dropout)
        else:
            entity_embeds = self.entity_embeds

        curvature, entity_weights, relation_weights = self.how1_enhancement(
            entity_embeds,
            self.relation_embeds,
            self._current_epoch,
            task_grad=entity_task_grad,
            curvature_grad=curvature_grad,
            edge_index=kg[0] if kg is not None else None,
            edge_type=kg[1] if kg is not None else None
        )
        self.current_entity_weights = entity_weights
        self.current_relation_weights = relation_weights

        dim_h = self.latent_dim // 3
        dim_s = self.latent_dim // 3
        dim_e = self.latent_dim - 2 * dim_h
        
        h_part = PoincareBall.project(PoincareBall.expmap0(entity_embeds[:, :dim_h])) * entity_weights[:, 0:1]
        s_part = SphericalManifold.project(entity_embeds[:, dim_h:dim_h+dim_s]) * entity_weights[:, 1:2]
        e_part = entity_embeds[:, dim_h+dim_s:] * entity_weights[:, 2:3]
        entity_embeds = torch.cat([h_part, s_part, e_part], dim=-1)
        self.current_product_entity_embeds = entity_embeds
        self.current_curvature = curvature

        kg_edge_index = kg[0] if kg is not None else None
        kg_edge_type = kg[1] if kg is not None else None
        entity_embeds, depths, depths_int, curv, sat_stats = self.how2_enhancement(
            entity_embeds,
            edge_index=kg_edge_index,
            edge_type=kg_edge_type,
            relation_embeds=self.relation_embeds,
            precomputed={'curvature': curvature},
            task_grad=entity_task_grad,
            curvature_grad=curvature_grad
        )
        self.current_diffused_entity_embeds = entity_embeds
        self.how2_stats = {
            'depths': depths,
            'depths_int': depths_int,
            'curvature': curv,
            'saturation_stats': sat_stats,
            'layer_entity_counts': sat_stats.get('layer_entity_counts', [])
        }

        item_embeds = entity_embeds[:self.item_num]
        embeds = torch.cat([self.user_embeds, item_embeds], dim=0)

        embeds_list = [embeds]
        for gcn in self.gcn_layers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        
        rec_curvature = self._entity_signal_to_rec_nodes(curv)
        entity_degrees = self.how1_enhancement.curvature_calculator.entity_degrees
        degree_norm = entity_degrees / torch.clamp(entity_degrees.max(), min=1.0)
        rec_degrees = self._entity_signal_to_rec_nodes(degree_norm)
        self.current_rec_curvature = rec_curvature
        self.current_rec_degrees = rec_degrees


        if compute_alignment:
            how3_weights = self.task_alignment(
                embeds,
                task_grad=task_grad,
                bakry_emery_curvature=rec_curvature,
                degree_norm=rec_degrees
            )
            self.how3_weights_cache = how3_weights
        else:
            self.how3_weights_cache = None
        
        return embeds[:self.user_num], embeds[self.user_num:]
    

    def set_training_phase(self, epoch: int):
        """Two-stage protocol: freeze curvature/alignment during warm-up, unfreeze later."""
        is_warmup = epoch < self.warmup_epochs
        for param in self.task_alignment.parameters():
            param.requires_grad = not is_warmup
        for module in [self.unified_curvature, self.how1_enhancement.curvature_calculator, self.how2_enhancement.curvature_calculator]:
            module.curvature_eta.requires_grad = not is_warmup
            module.be_potential.requires_grad = not is_warmup

    def set_current_epoch(self, epoch: int):
        """Set the current training epoch."""
        self._current_epoch = epoch
        self.set_training_phase(epoch)
        if self.task_alignment is not None:
            self.task_alignment.set_current_epoch(epoch)


# Backward compatibility for old scripts.
KnowledgeGraphRecommendationModel = HiCAPM
