#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HiCAPM main program.

Note: The entire model architecture remains in model.py. This file is
responsible only for configuration, data handling, training, and evaluation.
"""

import argparse
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

try:
    import geoopt
except ImportError:
    geoopt = None

from model import HiCAPM


# =============================================================================
# 1. config
# =============================================================================


@dataclass
class Config:
    data: str = None
    data_dir: str = None
    latent_dim: int = 64
    n_layers: int = 3
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 0.0001
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    seed: int = 42
    eval_interval: int = 1


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")


def available_datasets() -> List[str]:
    if os.path.isdir(DATASETS_ROOT):
        return sorted(
            name for name in os.listdir(DATASETS_ROOT)
            if os.path.isdir(os.path.join(DATASETS_ROOT, name))
            and not name.startswith(".")
            and name != "__pycache__"
        )
    return []

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiCAPM recommendation system")
    parser.add_argument("--data", type=str, default=None, help="Dataset name under ./datasets")
    parser.add_argument("--data_dir", type=str, default=None, help="Custom dataset directory")
    parser.add_argument("--latdim", "--latent_dim", dest="latent_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--list_datasets", action="store_true", help="List available datasets")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    data = args.data
    if data is None and args.data_dir is None:
        raise ValueError(
            "Specify a dataset under datasets/ with --data, or provide a "
            "custom dataset path with --data_dir"
        )
    if data is None:
        data = os.path.basename(os.path.abspath(args.data_dir.rstrip(os.sep))) or "custom"

    return Config(
        data=data,
        data_dir=args.data_dir,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        eval_interval=args.eval_interval,
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. data process
# =============================================================================


class InteractionDataset(torch.utils.data.Dataset):
    """User-item interaction dataset."""

    def __init__(self, interactions: List[Tuple[int, int]], num_items: int, is_test: bool = False):
        self.interactions = interactions
        self.num_items = num_items
        self.is_test = is_test
        self.user_items = defaultdict(set)
        for user, item in interactions:
            self.user_items[int(user)].add(int(item))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]
        if self.is_test:
            return torch.LongTensor([user]), torch.LongTensor([pos_item])

        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in self.user_items[user]:
            neg_item = random.randint(0, self.num_items - 1)

        return torch.LongTensor([user]), torch.LongTensor([pos_item]), torch.LongTensor([neg_item])


class DataModule:
    """Load and preprocess data, build adjacency matrices, and create data loaders."""

    def __init__(self, dataset_name: str = None, data_dir: str = None):
        self.dataset_name = dataset_name
        if data_dir is not None:
            self.data_dir = data_dir
        elif dataset_name is not None:
            self.data_dir = os.path.join(DATASETS_ROOT, dataset_name)
        else:
            raise ValueError("Specify a custom dataset path with --data_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.user_num = 0
        self.item_num = 0
        self.entity_num = 0
        self.relation_num = 0
        self.train_data = None
        self.test_data = None
        self.train_mat = None
        self.test_mat = None
        self.kg_triples = None
        self.kg_dict = defaultdict(list)
        self.kg_edges = []
        self.adj_matrix = None
        self.relation_dict = {}
        self.test_locs = defaultdict(list)

    def _normalize_adj(self, mat):
        import scipy.sparse as sp

        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _build_adjacency_matrix(self):
        try:
            import scipy.sparse as sp

            if self.train_mat is not None:
                trn_mat = self.train_mat
                user_block = sp.csr_matrix((self.user_num, self.user_num))
                item_block = sp.csr_matrix((self.item_num, self.item_num))
                mat = sp.vstack([sp.hstack([user_block, trn_mat]), sp.hstack([trn_mat.transpose(), item_block])])
                mat = (mat != 0) * 1.0
                mat = (mat + sp.eye(mat.shape[0])) * 1.0
                mat = self._normalize_adj(mat).tocoo()
                idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
                vals = torch.from_numpy(mat.data.astype(np.float32))
                self.adj_matrix = torch.sparse_coo_tensor(idxs, vals, torch.Size(mat.shape)).to(self.device)
                return
        except Exception as exc:
            print(f"Falling back to basic sparse adjacency matrix construction: {exc}")

        rows, cols = [], []
        for user, item in self.train_data:
            rows.extend([user, item + self.user_num])
            cols.extend([item + self.user_num, user])
        indices = torch.LongTensor([rows, cols])
        values = torch.FloatTensor([1.0] * len(rows))
        size = self.user_num + self.item_num
        self.adj_matrix = torch.sparse_coo_tensor(indices, values, (size, size)).to(self.device)

    def load_real_data(self):
        import scipy.sparse as sp
        from scipy.sparse import coo_matrix

        print(f"{datetime.now()}: Load Data")

        trn_file = os.path.join(self.data_dir, "trnMat.pkl")
        tst_file = os.path.join(self.data_dir, "tstMat.pkl")
        kg_file = os.path.join(self.data_dir, "kg.txt")

        if os.path.exists(trn_file) and os.path.exists(tst_file):
            with open(trn_file, "rb") as f:
                trn_mat = pickle.load(f)
            with open(tst_file, "rb") as f:
                tst_mat = pickle.load(f)

            if hasattr(trn_mat, "coords") and not hasattr(trn_mat, "row"):
                trn_mat = sp.coo_matrix((trn_mat.data, (trn_mat.coords[0], trn_mat.coords[1])), shape=trn_mat.shape)
            if hasattr(tst_mat, "coords") and not hasattr(tst_mat, "row"):
                tst_mat = sp.coo_matrix((tst_mat.data, (tst_mat.coords[0], tst_mat.coords[1])), shape=tst_mat.shape)

            trn_mat = (trn_mat != 0).astype(np.float32)
            tst_mat = (tst_mat != 0).astype(np.float32)
            self.train_mat = trn_mat if type(trn_mat) == coo_matrix else sp.coo_matrix(trn_mat)
            self.test_mat = tst_mat if type(tst_mat) == coo_matrix else sp.coo_matrix(tst_mat)
            self.user_num, self.item_num = self.train_mat.shape
            self.train_data = list(zip(self.train_mat.row.tolist(), self.train_mat.col.tolist()))
            self.test_data = list(zip(self.test_mat.row.tolist(), self.test_mat.col.tolist()))
        else:
            train_file = os.path.join(self.data_dir, "train.txt")
            test_file = os.path.join(self.data_dir, "test.txt")
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                raise FileNotFoundError(
                    "The dataset directory must contain trnMat.pkl/tstMat.pkl "
                    "or train.txt/test.txt"
                )
            self.train_data = self._load_interaction_txt(train_file)
            self.test_data = self._load_interaction_txt(test_file)
            self.user_num = max([u for u, _ in self.train_data + self.test_data]) + 1
            self.item_num = max([i for _, i in self.train_data + self.test_data]) + 1
            rows = [u for u, _ in self.train_data]
            cols = [i for _, i in self.train_data]
            self.train_mat = sp.coo_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(self.user_num, self.item_num),
            )

        kg_triplets = np.loadtxt(kg_file, dtype=np.int32)
        if kg_triplets.ndim == 1:
            kg_triplets = kg_triplets.reshape(1, -1)
        kg_triplets = np.unique(kg_triplets[:, :3], axis=0)
        inv_triplets = kg_triplets.copy()
        inv_triplets[:, 0] = kg_triplets[:, 2]
        inv_triplets[:, 2] = kg_triplets[:, 0]
        inv_triplets[:, 1] = kg_triplets[:, 1] + max(kg_triplets[:, 1]) + 1
        triplets = np.concatenate((kg_triplets, inv_triplets), axis=0)

        self.relation_num = int(max(triplets[:, 1]) + 1)
        self.entity_num = max(int(max(max(triplets[:, 0]), max(triplets[:, 2])) + 1), self.item_num)
        self.kg_dict = defaultdict(list)
        self.kg_edges = []
        kg_counter = defaultdict(set)
        for h_id, r_id, t_id in triplets:
            if t_id not in kg_counter[h_id]:
                kg_counter[h_id].add(t_id)
                self.kg_edges.append([int(h_id), int(t_id), int(r_id)])
                self.kg_dict[int(h_id)].append((int(r_id), int(t_id)))
        self.kg_triples = triplets.tolist()

        for head, neighbors in self.kg_dict.items():
            self.relation_dict[head] = {tail: relation for relation, tail in neighbors}
        for user, item in self.test_data:
            self.test_locs[user].append(item)

        self._build_adjacency_matrix()


        num_interactions = self.train_mat.nnz if self.train_mat is not None else len(self.train_data)

        print(f"kg shape:  ({self.entity_num}, {self.entity_num})")
        print(f"number of edges in KG:  {len(self.kg_edges)}")
        print(f"USER {self.user_num} ITEM {self.item_num}")
        print(f"NUM OF INTERACTIONS {num_interactions}")

    def _load_interaction_txt(self, file_path: str) -> List[Tuple[int, int]]:
        interactions = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = [int(x) for x in line.strip().split()]
                if len(values) < 2:
                    continue
                user, items = values[0], values[1:]
                interactions.extend((user, item) for item in items)
        return interactions

    def load_and_preprocess_data(self):
        self.load_real_data()
        return self.train_data, self.test_data, self.kg_triples

    def create_data_loaders(self, batch_size: int = 256):
        train_dataset = InteractionDataset(self.train_data, self.item_num)
        test_dataset = InteractionDataset(self.test_data, self.item_num, is_test=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, test_loader


def process_data(config: Config) -> DataModule:
    print("\n[1/4] data process")
    data_module = DataModule(dataset_name=config.data, data_dir=config.data_dir)
    data_module.load_and_preprocess_data()
    return data_module


# =============================================================================
# 3. model training
# =============================================================================


class TrainingModule:
    """HiCAPM training workflow."""

    def __init__(self, model: nn.Module, data_module: DataModule, learning_rate: float, weight_decay: float, device: str):
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.warmup_epochs = getattr(getattr(model, "task_alignment", None), "warmup_epochs", 10)
        self.reg_weight = weight_decay
        optimizer_cls = geoopt.optim.RiemannianAdam if geoopt is not None else torch.optim.Adam
        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.optimizer = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.evaluator = MultiMetricsEvaluator(k_list=[5, 10, 20])
        self.train_losses = []
        self.test_recalls = []
        self.best_metric_key = "Recall@20"
        self.best_metrics = {"epoch": 0, "score": float("-inf"), "metrics": {}}
        self.best_warmup_epoch = 0
        self.best_warmup_score = float("-inf")
        self.best_warmup_state = None

    def bpr_loss_per_sample(self, user_embeds, pos_item_embeds, neg_item_embeds):
        pos_scores = torch.sum(user_embeds * pos_item_embeds, dim=-1)
        neg_scores = torch.sum(user_embeds * neg_item_embeds, dim=-1)
        return -F.logsigmoid(pos_scores - neg_scores)

    def bpr_loss(self, user_embeds, pos_item_embeds, neg_item_embeds):
        return self.bpr_loss_per_sample(user_embeds, pos_item_embeds, neg_item_embeds).mean()

    def how1_loss(self, entity_weights, relation_weights):
        entity_std = entity_weights.std(dim=0).mean()
        relation_std = relation_weights.std(dim=0).mean()
        std_loss = -(entity_std + relation_std) / 2
        entity_entropy = -(entity_weights * torch.log(entity_weights + 1e-8)).sum(dim=1).mean()
        relation_entropy = -(relation_weights * torch.log(relation_weights + 1e-8)).sum(dim=1).mean()
        entropy_loss = -(entity_entropy + relation_entropy) / 2
        return 0.1 * (std_loss + entropy_loss)

    def how3_loss(self, how3_weights):
        if how3_weights is None:
            return torch.tensor(0.0, device=self.device)
        weights = how3_weights.squeeze()
        entropy = -torch.mean(weights * torch.log(weights + 1e-8) + (1 - weights) * torch.log(1 - weights + 1e-8))
        variance_loss = -torch.var(weights)
        return 0.1 * (entropy + variance_loss)
    def regularization_loss(self):
        reg = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            if param.requires_grad:
                reg = reg + torch.sum(param ** 2)
        return self.reg_weight * reg

    def _build_kg_tensor(self):
        kg_edges, edge_types = [], []
        for h, neighbors in self.data_module.kg_dict.items():
            for r, t in neighbors:
                kg_edges.append([h, t])
                edge_types.append(r)
        if not kg_edges:
            return None
        return torch.LongTensor(kg_edges).t().to(self.device), torch.LongTensor(edge_types).to(self.device)

    def _project_ball(self, x, eps: float = 1e-5):
        norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-8)
        return torch.where(norm > 1.0 - eps, x / norm * (1.0 - eps), x)

    def _product_manifold_distance(self, x, y):
        latent_dim = x.shape[-1]
        dim_h = latent_dim // 3
        dim_s = latent_dim // 3
        dim_e = latent_dim - 2 * dim_h
        distances = []
        if dim_h > 0:
            x_h = self._project_ball(x[:, :dim_h])
            y_h = self._project_ball(y[:, :dim_h])
            x2 = torch.sum(x_h * x_h, dim=-1)
            y2 = torch.sum(y_h * y_h, dim=-1)
            diff2 = torch.sum((x_h - y_h) ** 2, dim=-1)
            denom = torch.clamp((1.0 - x2) * (1.0 - y2), min=1e-8)
            distances.append(torch.acosh(torch.clamp(1.0 + 2.0 * diff2 / denom, min=1.0 + 1e-6)))
        if dim_s > 0:
            x_s = F.normalize(x[:, dim_h:dim_h + dim_s], p=2, dim=-1)
            y_s = F.normalize(y[:, dim_h:dim_h + dim_s], p=2, dim=-1)
            cosine = torch.sum(x_s * y_s, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            distances.append(torch.acos(cosine))
        if dim_e > 0:
            distances.append(torch.norm(x[:, dim_h + dim_s:] - y[:, dim_h + dim_s:], dim=-1))
        return torch.stack(distances, dim=0).sum(dim=0)

    def L_geo(self, kg, alpha_prime=None, margin: float = 1.0, max_triples: int = 4096):
        entity_embeds = getattr(self.model, "current_product_entity_embeds", None)
        if kg is None or entity_embeds is None:
            return torch.tensor(0.0, device=self.device)
        edge_index, _ = kg
        heads, tails = edge_index
        valid = (heads >= 0) & (heads < entity_embeds.shape[0]) & (tails >= 0) & (tails < entity_embeds.shape[0])
        heads, tails = heads[valid], tails[valid]
        if heads.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        if heads.numel() > max_triples:
            perm = torch.randperm(heads.numel(), device=self.device)[:max_triples]
            heads, tails = heads[perm], tails[perm]
        neg_tails = torch.randint(0, entity_embeds.shape[0], tails.shape, device=self.device)
        neg_tails = torch.where(neg_tails == tails, (neg_tails + 1) % entity_embeds.shape[0], neg_tails)
        pos_dist = self._product_manifold_distance(entity_embeds[heads], entity_embeds[tails])
        neg_dist = self._product_manifold_distance(entity_embeds[heads], entity_embeds[neg_tails])
        geo = F.relu(margin - pos_dist + neg_dist)
        if alpha_prime is not None:
            geo = alpha_prime[heads].to(geo.device) * geo
        return geo.mean()

    def _build_task_gradients(self, adj, kg, users, pos_items, neg_items):
        probe_users, probe_items = self.model(adj, kg=kg, compute_alignment=False)
        probe_loss = self.bpr_loss(probe_users[users], probe_items[pos_items], probe_items[neg_items])
        grad_targets = [probe_users, probe_items]
        curvature_target = getattr(self.model, "current_curvature", None)
        if curvature_target is not None and curvature_target.requires_grad:
            grad_targets.append(curvature_target)
        grads = torch.autograd.grad(probe_loss, grad_targets, allow_unused=True, retain_graph=False)
        user_grad = grads[0] if grads[0] is not None else torch.zeros_like(probe_users)
        item_grad = grads[1] if grads[1] is not None else torch.zeros_like(probe_items)
        rec_task_grad = torch.cat([user_grad.detach(), item_grad.detach()], dim=0)
        curvature_grad = grads[2].detach() if len(grads) > 2 and grads[2] is not None else None
        return rec_task_grad, curvature_grad

    def adaptive_total_loss(self, bpr_per_sample, users, pos_items, neg_items, epoch: int, kg=None):
        how3_weights = self.model.how3_weights_cache
        if how3_weights is None:
            how3_weights = torch.full((self.model.user_num + self.model.item_num, 1), 0.5, device=self.device)
        alpha_prime = how3_weights.squeeze(-1)
        user_nodes = users
        pos_nodes = pos_items + self.model.user_num
        neg_nodes = neg_items + self.model.user_num
        alpha_batch = (alpha_prime[user_nodes] + alpha_prime[pos_nodes] + alpha_prime[neg_nodes]) / 3.0
        task_loss = ((1.0 - alpha_batch) * bpr_per_sample).mean()
        alpha_entity = torch.full((self.model.entity_num,), alpha_prime.mean().detach(), device=self.device, dtype=alpha_prime.dtype)
        item_alpha = alpha_prime[self.model.user_num:self.model.user_num + self.model.item_num]
        limit = min(self.model.item_num, self.model.entity_num, item_alpha.shape[0])
        alpha_entity[:limit] = item_alpha[:limit]
        geo_loss = self.L_geo(kg, alpha_prime=alpha_entity)

        how1 = torch.tensor(0.0, device=self.device)
        if epoch >= self.warmup_epochs and self.model.current_entity_weights is not None:
            how1 = self.how1_loss(self.model.current_entity_weights, self.model.current_relation_weights)
        how3 = self.how3_loss(how3_weights) if epoch >= self.warmup_epochs else torch.tensor(0.0, device=self.device)
        reg = self.regularization_loss()
        return task_loss + geo_loss + how1 + how3 + reg, {
            "task_loss": task_loss,
            "geo_loss": geo_loss,
            "how1_loss": how1,
            "how3_loss": how3,
            "reg_loss": reg,
        }

    def train_epoch(self, train_loader, epoch: int):
        self.model.train()
        self.model.set_current_epoch(epoch)
        totals = defaultdict(float)
        kg = self._build_kg_tensor()
        if hasattr(self.model, "initialize_topology_priors"):
            self.model.initialize_topology_priors(kg)


        for batch in train_loader:
            users, pos_items, neg_items = batch
            users = users.squeeze().to(self.device)
            pos_items = pos_items.squeeze().to(self.device)
            neg_items = neg_items.squeeze().to(self.device)
            self.optimizer.zero_grad()

            rec_task_grad, curvature_grad = None, None
            if epoch >= self.warmup_epochs:
                rec_task_grad, curvature_grad = self._build_task_gradients(
                    self.data_module.adj_matrix, kg, users, pos_items, neg_items
                )
                ramp = min(1.0, max(0.0, (epoch - self.warmup_epochs + 1) / max(float(self.warmup_epochs), 1.0)))
                if rec_task_grad is not None:
                    rec_task_grad = rec_task_grad * ramp
                if curvature_grad is not None:
                    curvature_grad = curvature_grad * ramp
                self.optimizer.zero_grad()

            user_embeds, item_embeds = self.model(
                self.data_module.adj_matrix,
                kg=kg,
                task_grad=rec_task_grad,
                curvature_grad=curvature_grad,
            )
            bpr_per_sample = self.bpr_loss_per_sample(user_embeds[users], item_embeds[pos_items], item_embeds[neg_items])
            bpr = bpr_per_sample.mean()
            loss, parts = self.adaptive_total_loss(bpr_per_sample, users, pos_items, neg_items, epoch, kg=kg)
            loss.backward()
            self.optimizer.step()

            totals["loss"] += loss.item()
            totals["bpr_loss"] += bpr.item()
            for key, value in parts.items():
                totals[key] += value.item()
            totals["batches"] += 1

        batches = max(int(totals["batches"]), 1)
        avg = {key: value / batches for key, value in totals.items() if key != "batches"}
        self.train_losses.append(avg["loss"])
        return avg

    def evaluate(self, test_loader, epoch: int):
        self.model.eval()
        all_predictions, all_users = [], []
        kg = self._build_kg_tensor()
        if hasattr(self.model, "initialize_topology_priors"):
            self.model.initialize_topology_priors(kg)


        with torch.no_grad():
            user_embeds, item_embeds = self.model(self.data_module.adj_matrix, kg=kg, mess_dropout=False)
            for batch in test_loader:
                users, _ = batch
                users = users.squeeze().to(self.device)
                scores = torch.mm(user_embeds[users], item_embeds.t())
                _, top_items = torch.topk(scores, k=min(20, item_embeds.shape[0]), dim=-1)
                all_predictions.append(top_items.cpu().numpy())
                all_users.append(users.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_users = np.concatenate(all_users, axis=0)
        ground_truth = defaultdict(list)
        for user, item in self.data_module.test_data:
            ground_truth[user].append(item)
        metrics = self.evaluator.calculate_metrics(all_predictions, ground_truth, all_users)
        self.test_recalls.append(metrics.get("Recall@20", 0))
        current_epoch = epoch + 1
        score = metrics.get(self.best_metric_key, 0)
        if current_epoch <= self.warmup_epochs and score > self.best_warmup_score:
            self.best_warmup_epoch = current_epoch
            self.best_warmup_score = score
            self.best_warmup_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
        if score > self.best_metrics["score"]:
            self.best_metrics = {
                "epoch": current_epoch,
                "score": score,
                "metrics": dict(metrics),
            }
        return metrics

    def print_metrics_table(self, metrics: Dict[str, float], prefix: str = "  "):
        for k in self.evaluator.k_list:
            print(
                "%s@%d: Precision=%.4f, Recall=%.4f, F1=%.4f, HitRatio=%.4f, NDCG=%.4f, MAP=%.4f"
                % (
                    prefix,
                    k,
                    metrics.get("Precision@%d" % k, 0),
                    metrics.get("Recall@%d" % k, 0),
                    metrics.get("F1-Score@%d" % k, 0),
                    metrics.get("Hit Ratio@%d" % k, 0),
                    metrics.get("NDCG@%d" % k, 0),
                    metrics.get("MAP@%d" % k, 0),
                )
            )
    def _print_epoch_summary(self, epoch: int):
        if self.model.how2_stats is not None:
            depths = self.model.how2_stats["depths_int"]
            curvature = self.model.how2_stats["curvature"]
            unique_depths, counts = torch.unique(depths.detach().cpu(), return_counts=True)
            depth_text = ", ".join(f"{int(k.item())}: {int(v.item())}" for k, v in zip(unique_depths, counts))
            print(
                f"[How2] Epoch {epoch}: Depths={{{depth_text}}}, "
                f"Curvature(mean={curvature.float().mean().item():.4f}, "
                f"std={curvature.float().std().item():.4f})"
            )
        if self.model.current_entity_weights is not None:
            weights = self.model.current_entity_weights.mean(dim=0)
            print(f"Manifold Weights: H={weights[0].item():.4f}, S={weights[1].item():.4f}, E={weights[2].item():.4f}")
        if self.model.how3_weights_cache is not None:
            weights = self.model.how3_weights_cache.detach().float()
            print(
                f"[How3] Epoch {epoch}: Weights(mean={weights.mean().item():.4f}, "
                f"std={weights.std().item():.4f})"
            )

    def train(self, epochs: int, eval_interval: int, batch_size: int):
        print("Model Prepared")
        print("Model Initialized")
        train_loader, test_loader = self.data_module.create_data_loaders(batch_size=batch_size)
        for epoch in range(epochs):
            if epoch == 0:
                print("[How3] Epoch 0: Entering the learning stage")
            if epoch == self.warmup_epochs:
                if self.best_warmup_state is not None:
                    self.model.load_state_dict({name: value.to(self.device) for name, value in self.best_warmup_state.items()})
                    self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg_weight)
                    print(f"[Stage2] Restored best warm-up checkpoint from Epoch {self.best_warmup_epoch}")
                print("=" * 80)
                print(
                    f"[Stage 2] Epoch {epoch + 1}: Unfreezing How1 parameters "
                    "and enabling dynamic curvature optimization"
                )
                print("=" * 80)

            train_metrics = self.train_epoch(train_loader, epoch)
            self._print_epoch_summary(epoch)
            if (epoch + 1) % eval_interval == 0:
                test_metrics = self.evaluate(test_loader, epoch)
                print(
                    f"Epoch [{epoch + 1}/{epochs}] - "
                    f"Loss: {train_metrics['loss']:.4f}, "
                    f"Task: {train_metrics.get('task_loss', 0):.4f}, "
                    f"Geo: {train_metrics.get('geo_loss', 0):.4f}"
                )
                print("  Evaluation:")
                self.print_metrics_table(test_metrics, prefix="    ")

        print()
        print(
            f"Training complete! Best result selected by {self.best_metric_key}: "
            f"Epoch {self.best_metrics['epoch']}"
        )
        self.print_metrics_table(self.best_metrics["metrics"], prefix="  ")
        return self.best_metrics


def build_model(config: Config, data_module: DataModule, device: str) -> HiCAPM:
    print("\n[2/4] model")
    model = HiCAPM(
        user_num=data_module.user_num,
        item_num=data_module.item_num,
        entity_num=data_module.entity_num,
        relation_num=data_module.relation_num,
        latent_dim=config.latent_dim,
        n_layers=config.n_layers,
        device=device,
        warmup_epochs=config.warmup_epochs,
    )
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def train_model(config: Config, model: HiCAPM, data_module: DataModule, device: str) -> TrainingModule:
    print("\n[3/4] model training")
    trainer = TrainingModule(
        model=model,
        data_module=data_module,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
    )
    trainer.train(epochs=config.epochs, eval_interval=config.eval_interval, batch_size=config.batch_size)
    return trainer


# =============================================================================
# 4. evaluation
# =============================================================================


class MultiMetricsEvaluator:
    """Evaluate multiple metrics: Precision, Recall, F1, Hit Ratio, NDCG, and MAP."""

    def __init__(self, k_list: List[int] = None):
        self.k_list = k_list or [5, 10, 20]

    def calculate_metrics(self, predictions: np.ndarray, ground_truth: Dict[int, List[int]], user_ids: np.ndarray) -> Dict[str, float]:
        metrics = {}
        for k in self.k_list:
            use_k = min(k, predictions.shape[1])
            precision_sum = recall_sum = f1_sum = 0.0
            hit_sum = ndcg_sum = map_sum = 0.0
            valid_users = 0

            for i, user_id in enumerate(user_ids):
                true_items = set(ground_truth[int(user_id)]) if ground_truth[int(user_id)] else set()
                if not true_items:
                    continue
                valid_users += 1
                pred_items_ordered = predictions[i, :use_k].tolist()
                pred_items = set(pred_items_ordered)
                hits = len(pred_items & true_items)

                precision = hits / use_k
                recall = hits / len(true_items)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                hit = 1.0 if hits > 0 else 0.0

                dcg = sum(1 / np.log2(j + 2) for j, item in enumerate(pred_items_ordered) if item in true_items)
                idcg = sum(1 / np.log2(j + 2) for j in range(min(len(true_items), use_k)))
                ndcg = dcg / idcg if idcg > 0 else 0.0

                ap = 0.0
                hit_count = 0
                for j, item in enumerate(pred_items_ordered):
                    if item in true_items:
                        hit_count += 1
                        ap += hit_count / (j + 1)
                ap = ap / min(len(true_items), use_k)

                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                hit_sum += hit
                ndcg_sum += ndcg
                map_sum += ap

            denom = max(valid_users, 1)
            metrics[f"Precision@{k}"] = precision_sum / denom
            metrics[f"Recall@{k}"] = recall_sum / denom
            metrics[f"F1-Score@{k}"] = f1_sum / denom
            metrics[f"Hit Ratio@{k}"] = hit_sum / denom
            metrics[f"NDCG@{k}"] = ndcg_sum / denom
            metrics[f"MAP@{k}"] = map_sum / denom
        return metrics


def evaluate_model(config: Config, trainer: TrainingModule, data_module: DataModule) -> Dict[str, float]:
    print("\nEvaluation - Best Epoch Metrics")
    metrics = trainer.best_metrics.get("metrics", {})
    if not metrics:
        _, test_loader = data_module.create_data_loaders(batch_size=config.batch_size)
        metrics = trainer.evaluate(test_loader, epoch=config.epochs - 1)
    print(f"Best Epoch: {trainer.best_metrics['epoch']} (selected by {trainer.best_metric_key})")
    trainer.print_metrics_table(metrics, prefix="  ")
    return metrics


def run_pipeline(config: Config):
    print("=" * 70)
    print("HiCAPM - Hierarchical Curvature-Adaptive Product Manifold")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_module = process_data(config)
    model = build_model(config, data_module, device)
    trainer = train_model(config, model, data_module, device)
    metrics = evaluate_model(config, trainer, data_module)
    print("\nDone.")
    return trainer, metrics


def main():
    args = parse_args()
    if args.list_datasets:
        print("\n".join(available_datasets()))
        return
    config = build_config(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
