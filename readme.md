# Graph Entropy Minimization for Semi-supervised Node Classification

This repo contains code that are required to reporduce all experiments in our paper *Graph Entropy Minimization for Semi-supervised Node Classification*.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-entropy-minimization-for-semi/node-classification-on-citeseer-with-public)](https://paperswithcode.com/sota/node-classification-on-citeseer-with-public?p=graph-entropy-minimization-for-semi)

## Paper Abstract

Node classifiers are required to comprehensively reduce prediction error, training resources, and inference latency in the industry.
However, most graph neural networks (GNN) concentrate only on one or two of them.
The compromised aspects thus are the shortest boards on the bucket, hindering their practical deployments for industrial-level tasks.
This work proposes a semi-supervised learning method termed Graph Entropy Minimization (GEM) to resolve the three issues simultaneously.
GEM benefits its one-hop aggregation from massive uncategorized nodes, making its prediction accuracy comparable to GNNs with two- or more-hops message passing.
It can be decomposed to support stochastic training with mini-batches of independent edge samples, achieving extremely fast sampling and space-saving training.
While its one-hop aggregation is faster in inference than deep GNNs, GEM can be furtherly accelerated to an extreme by deriving a non-hop classifier via online knowledge distillation.
Thus, GEM can be a handy choice for latency-restricted and error-sensitive services running on resource-constraint hardware.

## Usage

```
usage: main.pyc [-h] [--runs RUNS] [--gpu GPU] [--split SPLIT]
                [--test-bins TEST_BINS] [--inductive] [--lr LR]
                [--dropout DROPOUT] [--n-layers N_LAYERS]
                [--batch-size BATCH_SIZE] [--weight-decay WEIGHT_DECAY]
                [--early-stop EARLY_STOP] [--max-epochs MAX_EPOCHS]
                [--hidden HIDDEN] [--symmetric] [--er ER] [--gem GEM]
                [--kd KD] [--heads HEADS] [--alpha ALPHA] [--beta BETA]
                [--precompute PRECOMPUTE] [--correct CORRECT]
                [--correct-rate CORRECT_RATE] [--smooth SMOOTH]
                [--smooth-rate SMOOTH_RATE]
                method dataset

positional arguments:
  method                MLP | GCN | GIN | SAGE | GAT | GCNII | JKNet | ...
  dataset               cora | citeseer | pubmed | flickr | arxiv | yelp |
                        reddit | ...

options:
  -h, --help            show this help message and exit
  --runs RUNS           Default: 1
  --gpu GPU             Default: 0
  --split SPLIT         number of nodes per class for training. Set to 0 to
                        use default split (if any) or 20.
  --test-bins TEST_BINS
                        With distance from the training set larger than this
                        are tail nodes. Default: disabled
  --inductive           Enable the inductive setting.
  --lr LR               Learning Rate. Default: 0.01
  --dropout DROPOUT     Default: 0.5
  --n-layers N_LAYERS   Default: 2
  --batch-size BATCH_SIZE
                        Default: 0 (full batch)
  --weight-decay WEIGHT_DECAY
                        Default: 0
  --early-stop EARLY_STOP
                        Maximum iterations to stop when accuracy decreasing.
                        Default: 100
  --max-epochs MAX_EPOCHS
                        Maximum epochs. Default: 1000
  --hidden HIDDEN       Dimension of hidden representations and implicit
                        state. Default: 64
  --symmetric           Whether to symmetrically normalize adjacency matrix
  --er ER               Scale of MR loss. Default: 0.0
  --gem GEM             Scale of GEM in loss. Default: 0.0
  --kd KD               Offline knowledge distillation to MLP
  --heads HEADS         Number of attention heads for GAT/GEM. Default: 0
  --alpha ALPHA         alpha for GCNII or tau for GraphMLP/GEM. Default: 0.5
  --beta BETA           theta for GCNII or alpha for GraphMLP or for LinkDist.
                        Default: 1.0
  --precompute PRECOMPUTE
                        Precompute times for GraphMLP. Default: 2
  --correct CORRECT     Iterations for Correct after prediction. Default: 0
  --correct-rate CORRECT_RATE
                        Propagation rate for Correct after prediction.
                        Default: 0.0
  --smooth SMOOTH       Iterations for Smooth after prediction. Default: 0
  --smooth-rate SMOOTH_RATE
                        Propagation rate for Smooth after prediction. Default:
                        0.0
```

For example, if you want to run MLP on the Cora dataset with its default split on gpu `cuda:3` for 5 runs, execute

```bash
python3 main.py MLP cora --split 0 --gpu 3 --runs 5
```

## Reproducibility

Files in `scripts/` folder are scripts that reproduce experiments in our article.

* `run_baseline` runs experiments to produces accuracy scores for GEM, EEM, OKDEEM, and baselines in Table 2
* `run_hops` runs experiment and produces data for Figure 1.
* `run_labels` runs experiment to produce data for Figure 2.

## Datasets

Datasets used in our paper are retrieved with [DGL](https://github.com/dmlc/dgl), [PyG](https://github.com/pyg-team/pytorch_geometric), and [OGB](https://github.com/snap-stanford/ogb).

## Citation

```bibtex
@misc{luo2023graph,
      title={Graph Entropy Minimization for Semi-supervised Node Classification}, 
      author={Yi Luo and Guangchun Luo and Ke Qin and Aiguo Chen},
      year={2023},
      eprint={2305.19502},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
