
# HiCAPM


The implementation is organized around two files:

- `main.py`: configuration, data loading, training, evaluation, and command-line entry point.
- `model.py`: HiCAPM model components, including curvature estimation, product manifold weighting, asynchronous diffusion, and task alignment.

## Requirements

The code was developed with PyTorch 2.1.2. CUDA is recommended for full experiments.

Install dependencies:
```bash
pip install torch==2.1.2 numpy scipy geoopt>=0.5.0

```
When Geoopt is installed, training uses `geoopt.optim.RiemannianAdam`; otherwise it falls back to `torch.optim.Adam`.

## Quick Start

All datasets are loaded from the project `datasets/` directory.

List available datasets:

```bash
python main.py --list_datasets
```

Train on a dataset placed in `datasets/<dataset_name>/`:

```bash
python main.py --data <dataset_name> --epochs 30
```

## Main Options

```bash
python main.py \
  --data <dataset_name> \
  --latent_dim 64 \
  --n_layers 3 \
  --epochs 30 \
  --batch_size 1024 \
  --lr 0.0001 \
  --weight_decay 1e-4 \
  --warmup_epochs 10 \
  --eval_interval 1 \
  --seed 42
```

Available arguments:

- `--data`: dataset name under `datasets/`; required unless `--data_dir` is provided.
- `--data_dir`: custom dataset directory.
- `--latdim`, `--latent_dim`: embedding dimension.
- `--n_layers`: number of user-item GCN layers.
- `--epochs`: number of training epochs.
- `--batch_size`: mini-batch size.
- `--lr`, `--learning_rate`: learning rate.
- `--weight_decay`: L2 regularization weight.
- `--warmup_epochs`: number of geometric warm-up epochs.
- `--eval_interval`: evaluation interval in epochs.
- `--seed`: random seed.

## Model Components

The code implements the following components:

- **Dynamic curvature initialization and evolution**: `BakryEmeryCurvatureCalculator` builds topology priors from KG structure and updates curvature with task-gradient feedback.
- **Adaptive product manifold projection**: `How1HierarchicalCurvatureWeighting` generates hyperbolic, spherical, and Euclidean mixing weights for entity representations.
- **Curvature-depth mapping**: `CurvatureDepthMapper` maps dynamic curvature to node-specific propagation depth in `[1, 6]` using a sigmoid-modulated rule.
- **Curvature-aware asynchronous diffusion**: `How2AsyncDiffusion` propagates each node only up to its learned depth and adjusts neighbor attention by relative curvature gaps.
- **Node-level task alignment**: `TaskAlignmentModule` produces `alpha_prime(v)` from embeddings, curvature, degree information, and task-gradient feedback.
- **Adaptive objective**: `TrainingModule.adaptive_total_loss` combines node-weighted BPR task loss, product-manifold KG margin loss, How1/How3 regularization, and L2 regularization.


## Evaluation

The evaluator reports multiple top-K metrics for `K = 5, 10, 20`:

- Precision
- Recall
- F1-Score
- Hit Ratio
- NDCG
- MAP

Example output:

```text
Evaluation:
  @5: Precision=..., Recall=..., F1=..., HitRatio=..., NDCG=..., MAP=...
  @10: Precision=..., Recall=..., F1=..., HitRatio=..., NDCG=..., MAP=...
  @20: Precision=..., Recall=..., F1=..., HitRatio=..., NDCG=..., MAP=...
```

Keep:

- `main.py`
- `model.py`
- This `README.md`
