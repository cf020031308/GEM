import os
import time
import json
import copy
import hashlib
import datetime
import argparse

import psutil
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from dgl import data as dgl_data
from torch_geometric import datasets as pyg_data
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, default='MLP', help=(
    'MLP | GCN | GIN | SAGE | GAT | GCNII | JKNet | ... '
))
parser.add_argument('dataset', type=str, default='cora', help=(
    'cora | citeseer | pubmed | flickr | arxiv | yelp | reddit | ...'
))
parser.add_argument('--runs', type=int, default=1, help='Default: 1')
parser.add_argument('--gpu', type=int, default=0, help='Default: 0')
parser.add_argument(
    '--split', type=float, default=0,
    help=('number of nodes per class for training.'
          ' Set to 0 to use default split (if any) or 20.'))
parser.add_argument(
    '--test-bins', type=int, default=0, help=(
        'With distance from the training set '
        'larger than this are tail nodes. Default: disabled'))
parser.add_argument(
    '--inductive', action='store_true',
    help='Enable the inductive setting.')
parser.add_argument(
    '--lr', type=float, default=0.01, help='Learning Rate. Default: 0.01')
parser.add_argument(
    '--dropout', type=float, default=0.5, help='Default: 0.5')
parser.add_argument('--n-layers', type=int, default=2, help='Default: 2')
parser.add_argument(
    '--batch-size', type=int, default=0, help='Default: 0 (full batch)')
parser.add_argument(
    '--weight-decay', type=float, default=0.0, help='Default: 0')
parser.add_argument(
    '--early-stop', type=int, default=100,
    help='Maximum iterations to stop when accuracy decreasing. Default: 100')
parser.add_argument(
    '--max-epochs', type=int, default=1000,
    help='Maximum epochs. Default: 1000')
parser.add_argument(
    '--hidden', type=int, default=256,
    help='Dimension of hidden representations and implicit state. Default: 64')
parser.add_argument(
    '--symmetric', action='store_true',
    help='Whether to symmetrically normalize adjacency matrix')
parser.add_argument(
    '--er', type=float, default=0.0,
    help='Scale of MR loss. Default: 0.0')
parser.add_argument(
    '--gem', type=float, default=0.0,
    help='Scale of GEM in loss. Default: 0.0')
parser.add_argument(
    '--kd', type=float, default=0.0,
    help='Offline knowledge distillation to MLP')
parser.add_argument(
    '--heads', type=int, default=0,
    help='Number of attention heads for GAT/GEM. Default: 0')
parser.add_argument(
    '--alpha', type=float, default=0.5,
    help='alpha for GCNII or tau for GraphMLP/GEM. Default: 0.5')
parser.add_argument(
    '--beta', type=float, default=1.0,
    help='theta for GCNII or alpha for GraphMLP or for OKDEEM. Default: 1.0')
parser.add_argument(
    '--precompute', type=int, default=2,
    help='Precompute times for GraphMLP. Default: 2')
parser.add_argument(
    '--correct', type=int, default=0,
    help='Iterations for Correct after prediction. Default: 0')
parser.add_argument(
    '--correct-rate', type=float, default=0.0,
    help='Propagation rate for Correct after prediction. Default: 0.0')
parser.add_argument(
    '--smooth', type=int, default=0,
    help='Iterations for Smooth after prediction. Default: 0')
parser.add_argument(
    '--smooth-rate', type=float, default=0.0,
    help='Propagation rate for Smooth after prediction. Default: 0.0')
args = parser.parse_args()

has_cold = (
    args.gem and args.batch_size and args.beta
    or args.method == 'OKDEEM')
if not torch.cuda.is_available():
    args.gpu = -1
print(datetime.datetime.now(), args)
script_time = time.time()

g_dev = None
gpu = lambda x: x
if args.gpu >= 0:
    g_dev = torch.device('cuda:%d' % args.gpu)
    gpu = lambda x: x.to(g_dev)
coo = torch.sparse_coo_tensor
get_score = lambda y_true, y_pred: f1_score(
    y_true.cpu(), y_pred.cpu(), average='micro').item()
need_save = args.method in ('MLP', 'SAGE') and args.gem == 0.0


class Optim(object):
    def __init__(self, params):
        self.params = params
        self.opt = torch.optim.Adam(
            self.params, lr=args.lr, weight_decay=args.weight_decay)

    def extend(self, params):
        self.params.extend(params)
        self.opt = torch.optim.Adam(
            self.params, lr=args.lr, weight_decay=args.weight_decay)

    def __repr__(self):
        return 'params: %d' % sum(p.numel() for p in self.params)

    def __enter__(self):
        self.opt.zero_grad()
        self.elapsed = time.time()
        return self.opt

    def __exit__(self, *vs, **kvs):
        self.opt.step()
        self.elapsed = time.time() - self.elapsed


class JKNet(nn.Module):
    def __init__(self, din, dout, hidden, n_layers, dropout=0, **kw):
        super(self.__class__, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(gnn.GCNConv(din, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(gnn.GCNConv(hidden, hidden))
        self.lin = nn.Linear(hidden * n_layers, n_labels)
        self.jk = gnn.JumpingKnowledge(mode='cat')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        xs = []
        for conv in self.convs:
            x = F.relu(conv(self.dropout(x), edge_index))
            xs.append(x)
        return self.lin(self.jk(xs))


class GCNII(nn.Module):
    def __init__(
            self, din, dout, hidden, n_layers, dropout=0, **kw):
        super(self.__class__, self).__init__()
        self.lin1 = nn.Linear(din, hidden)
        self.convs = nn.ModuleList([
            gnn.GCN2Conv(
                channels=hidden,
                alpha=kw['alpha'],
                theta=kw['beta'],
                layer=i + 1,
            ) for i in range(n_layers)])
        self.lin2 = nn.Linear(hidden, dout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x0 = x = F.relu(self.lin1(self.dropout(x)))
        for conv in self.convs:
            x = F.relu(conv(self.dropout(x), x0, edge_index))
        return self.lin2(self.dropout(x))


def load_data(name):
    is_bidir = None
    if args.dataset in ('arxiv', 'mag', 'products'):
        ds = NodePropPredDataset(name='ogbn-%s' % args.dataset)
        train_idx, valid_idx, test_idx = map(
            ds.get_idx_split().get, 'train valid test'.split())
        if args.dataset == 'mag':
            train_idx = train_idx['paper']
            valid_idx = valid_idx['paper']
            test_idx = test_idx['paper']
        g, labels = ds[0]
        if args.dataset == 'mag':
            labels = labels['paper']
            g['edge_index'] = g['edge_index_dict'][('paper', 'cites', 'paper')]
            g['node_feat'] = g['node_feat_dict']['paper']
        X = torch.from_numpy(g['node_feat'])
        Y = torch.from_numpy(labels).clone().squeeze(-1)
        E = torch.from_numpy(g['edge_index'])
        n_nodes = X.shape[0]
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True
        is_bidir = False
        train_masks = [train_mask] * args.runs
        valid_masks = [valid_mask] * args.runs
        test_masks = [test_mask] * args.runs
    elif args.dataset in (
        'cora', 'citeseer', 'pubmed', 'corafull', 'reddit',
        'coauthor-cs', 'coauthor-phy', 'amazon-com', 'amazon-photo',
    ):
        g = (
            dgl_data.CoraGraphDataset() if args.dataset == 'cora'
            else dgl_data.CiteseerGraphDataset() if args.dataset == 'citeseer'
            else dgl_data.PubmedGraphDataset() if args.dataset == 'pubmed'
            else dgl_data.CoraFullDataset() if args.dataset == 'corafull'
            else dgl_data.RedditDataset() if args.dataset == 'reddit'
            else dgl_data.CoauthorCSDataset()
            if args.dataset == 'coauthor-cs'
            else dgl_data.CoauthorPhysicsDataset()
            if args.dataset == 'coauthor-phy'
            else dgl_data.AmazonCoBuyComputerDataset()
            if args.dataset == 'amazon-com'
            else dgl_data.AmazonCoBuyPhotoDataset()
            if args.dataset == 'amazon-photo'
            else None
        )[0]
        X, Y, train_mask, valid_mask, test_mask = map(
            g.ndata.get, 'feat label train_mask val_mask test_mask'.split())
        E = torch.cat([e.view(1, -1) for e in g.edges()], dim=0)
        is_bidir = True
        train_masks = [train_mask] * args.runs
        valid_masks = [valid_mask] * args.runs
        test_masks = [test_mask] * args.runs
    else:
        dn = 'dataset/' + args.dataset
        g = (
            pyg_data.Flickr(dn) if args.dataset == 'flickr'
            else pyg_data.Yelp(dn) if args.dataset == 'yelp'
            else pyg_data.AmazonProducts(dn) if args.dataset == 'amazon'
            else pyg_data.Actor(dn) if args.dataset == 'actor'
            else pyg_data.WebKB(dn, args.dataset.capitalize())
            if args.dataset in ('cornell', 'texas', 'wisconsin')
            else pyg_data.WikipediaNetwork(dn, args.dataset)
            if args.dataset in ('chameleon', 'crocodile', 'squirrel')
            else None
        ).data
        X, Y, E, train_mask, valid_mask, test_mask = map(
            g.get, 'x y edge_index train_mask val_mask test_mask'.split())
        if args.dataset in ('flickr', 'yelp', 'amazon'):
            train_masks = [train_mask] * args.runs
            valid_masks = [valid_mask] * args.runs
            test_masks = [test_mask] * args.runs
            is_bidir = True
        else:
            train_masks = [train_mask[:, i % train_mask.shape[1]]
                           for i in range(args.runs)]
            valid_masks = [valid_mask[:, i % train_mask.shape[1]]
                           for i in range(args.runs)]
            test_masks = [test_mask[:, i % train_mask.shape[1]]
                          for i in range(args.runs)]
            is_bidir = False
    if is_bidir is None:
        for i in range(E.shape[1]):
            src, dst = E[:, i]
            if src.item() != dst.item():
                print(src, dst)
                break
        is_bidir = ((E[0] == dst) & (E[1] == src)).any().item()
        print('guess is bidir:', is_bidir)
    n_labels = int(Y.max().item() + 1)
    # Save Label Transitional Matrices
    fn = 'dataset/labeltrans/%s.json' % args.dataset
    if not os.path.exists(fn):
        with open(fn, 'w') as file:
            mesh = coo(
                Y[E], torch.ones(E.shape[1]), size=(n_labels, n_labels)
            ).to_dense()
            den = mesh.sum(dim=1, keepdim=True)
            mesh /= den
            mesh[den.squeeze(1) == 0] = 0
            json.dump(mesh.tolist(), file)
    # Remove Self-Loops
    E = E[:, E[0] != E[1]]
    # Get Undirectional Edges
    if not is_bidir:
        E = torch.cat((E, E[[1, 0]]), dim=1)
    nrange = torch.arange(X.shape[0])
    if train_mask is None and not args.split:
        args.split = 0.6
    if 0 < args.split < 1:
        torch.manual_seed(42)  # the answer
        train_masks, valid_masks, test_masks = [], [], []
        for _ in range(args.runs):
            train_mask = torch.zeros(X.shape[0], dtype=bool)
            valid_mask = torch.zeros(X.shape[0], dtype=bool)
            test_mask = torch.zeros(X.shape[0], dtype=bool)
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
            for c in range(n_labels):
                label_idx = nrange[Y == c]
                val_num = test_num = int(
                    (1 - args.split) / 2 * label_idx.shape[0])
                perm = label_idx[torch.randperm(label_idx.shape[0])]
                train_mask[perm[val_num + test_num:]] = True
                valid_mask[perm[:val_num]] = True
                test_mask[perm[val_num:val_num + test_num]] = True
    elif int(args.split):
        torch.manual_seed(42)  # the answer
        train_masks, valid_masks, test_masks = [], [], []
        for _ in range(args.runs):
            train_mask = torch.zeros(X.shape[0], dtype=bool)
            for y in range(n_labels):
                label_mask = Y == y
                train_mask[
                    nrange[label_mask][
                        torch.randperm(label_mask.sum())[:int(args.split)]]
                ] = True
            valid_mask = ~train_mask
            valid_mask[
                nrange[valid_mask][torch.randperm(valid_mask.sum())[500:]]
            ] = False
            test_mask = ~(train_mask | valid_mask)
            test_mask[
                nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]
            ] = False
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
    # Split nodes by distance from the training set
    if args.test_bins:
        hop_bins = []
        for train_mask in train_masks:
            last_colored = train_mask
            hopbins = []
            for _ in range(args.test_bins):
                colored = last_colored.clone()
                ls, rs = last_colored[E]
                colored[E[1, ls]] = True
                colored[E[0, rs]] = True
                hopbins.append(~train_mask & colored & ~last_colored)
                last_colored = colored
            hopbins.append(~train_mask & ~last_colored)
            hop_bins.append(hopbins)
        for i in range(args.runs):
            print('hop bins %d:' % i, ', '.join(
                '%d (%.2f%%)' % (
                    hopbins.sum(), 100 * hopbins.sum() / X.shape[0])
                for hopbins in hop_bins[i]))
        valid_masks, test_masks = [], []
        torch.manual_seed(42)  # the answer
        for train_mask, hopbins in zip(train_masks, hop_bins):
            test_bins = []
            valid_mask = ~train_mask
            for hopbin in hopbins:
                testbin = torch.zeros(X.shape[0], dtype=bool)
                testbin[
                    nrange[hopbin][torch.randperm(hopbin.sum())][:100]] = True
                test_bins.append(testbin)
                valid_mask[testbin] = False
            valid_mask[
                nrange[valid_mask][torch.randperm(valid_mask.sum())][500:]
            ] = False
            valid_masks.append(valid_mask)
            test_masks.append(test_bins)
    return X, Y, E, train_masks, valid_masks, test_masks, is_bidir


class Stat(object):
    def __init__(self):
        self.preprocess_time = 0
        self.training_times = []
        self.evaluation_times = []

        self.best_test_scores = []
        self.best_times = []
        self.best_training_times = []
        self.best_test_scores_trans = []
        self.best_times_trans = []
        self.best_training_times_trans = []

        self.mem = psutil.Process().memory_info().rss / 1024 / 1024
        self.gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            self.gpu = torch.cuda.memory_allocated(g_dev) / 1024 / 1024

    def start_preprocessing(self):
        self.preprocess_time = time.time()

    def stop_preprocessing(self):
        self.preprocess_time = time.time() - self.preprocess_time

    def start_run(self):
        self.params = None
        self.scores = []
        self.scores_trans = []
        self.acc_training_times = []
        self.acc_times = []
        self.iterations = 0
        self.training_times.append(0.)
        self.evaluation_times.append(0.)

    def record_training(self, elapsed):
        self.iterations += 1
        self.training_times[-1] += elapsed

    def record_evaluation(self, elapsed):
        self.evaluation_times[-1] += elapsed

    def evaluate_result(self, y):
        if has_cold:
            y, ay = y
        scores = [get_score(Y[valid_mask], y[valid_mask].argmax(dim=1))]
        if args.test_bins:
            scores.append([
                get_score(Y[testbin], y[testbin].argmax(dim=1))
                if testbin.sum().item() else 0
                for testbin in test_mask])
        else:
            scores.append(get_score(Y[test_mask], y[test_mask].argmax(dim=1)))
        self.scores.append(scores)
        if has_cold:
            scores_trans = [
                get_score(Y[valid_mask], ay[valid_mask].argmax(dim=1))]
            if args.test_bins:
                scores_trans.append([
                    get_score(Y[testbin], ay[testbin].argmax(dim=1))
                    if testbin.sum().item() else 0
                    for testbin in test_mask])
            else:
                scores_trans.append(
                    get_score(Y[test_mask], ay[test_mask].argmax(dim=1)))
            self.scores_trans.append(scores_trans)
        self.acc_training_times.append(self.training_times[-1])
        self.acc_times.append(
            self.preprocess_time
            + self.training_times[-1])
        dec_epochs = len(self.scores) - 1 - torch.tensor([
            s for s, _ in self.scores]).argmax()
        if dec_epochs == 0:
            self.best_acc = self.scores[-1][0]
            self.best_y = y
            self.iterations = 0
        if args.batch_size:
            return self.iterations >= args.early_stop and dec_epochs >= 3
        return dec_epochs >= args.early_stop

    def end_run(self):
        val_scores = [s for s, _ in self.scores]
        print('val scores:', val_scores)
        val_score, idx = torch.tensor(val_scores).max(0)
        print('best valid score:', idx, val_score.item())
        if not args.test_bins:
            print('test scores:', [s for _, s in self.scores])
        print('acc training times:', self.acc_training_times)
        self.best_test_scores.append((idx, self.scores[idx][-1]))
        self.best_training_times.append(self.acc_training_times[idx])
        self.best_times.append(self.acc_times[idx])
        print('best test score:', self.best_test_scores[-1])
        if has_cold:
            val_scores = [s for s, _ in self.scores_trans]
            print('val scores (cold):', val_scores)
            val_score, idx = torch.tensor(val_scores).max(0)
            print('best valid score (cold):', idx, val_score.item())
            if not args.test_bins:
                print('test scores (cold):',
                      [s for _, s in self.scores_trans])
            self.best_test_scores_trans.append((
                idx, self.scores_trans[idx][-1]))
            self.best_training_times_trans.append(self.acc_training_times[idx])
            self.best_times_trans.append(self.acc_times[idx])
            print('best test score (cold):', self.best_test_scores_trans[-1])
        if need_save:
            try:
                wocs = copy.deepcopy(args)
                wocs.runs = 0
                prs_fn = 'predictions/%s-%s-%s.json' % (
                    args.method,
                    args.dataset,
                    hashlib.md5(str(wocs).encode()).hexdigest())
                if os.path.exists(prs_fn):
                    with open(prs_fn) as file:
                        prs = json.load(file)
                else:
                    prs = {}
                prs['settings'] = str(wocs)
                prs['date'] = str(datetime.datetime.now())
                prs['run-%d' % run] = [
                    self.best_acc, self.best_y.cpu().tolist()]
                with open(prs_fn, 'w') as file:
                    json.dump(prs, file)
            except Exception:
                print('Exception: failed to save predictions.')

    def end_all(self):
        conv = 1.0 + torch.tensor([
            idx for idx, _ in self.best_test_scores])
        score = 100 * torch.tensor([
            score for _, score in self.best_test_scores])
        tm = torch.tensor(self.best_times)
        ttm = torch.tensor(self.best_training_times)
        print('converge time: %.3f±%.3f' % (
            tm.mean().item(), tm.std().item()))
        print('converge training time: %.3f±%.3f' % (
            ttm.mean().item(), ttm.std().item()))
        print('converge epochs: %.3f±%.3f' % (
            conv.mean().item(), conv.std().item()))
        if args.test_bins:
            print('score:', ', '.join(
                '%.2f±%.2f' % s
                for s in zip(score.mean(dim=0), score.std(dim=0))))
        else:
            print('score: %.2f±%.2f' % (
                score.mean().item(), score.std().item()))
        if has_cold:
            conv = 1.0 + torch.tensor([
                idx for idx, _ in self.best_test_scores_trans])
            score = 100 * torch.tensor([
                score for _, score in self.best_test_scores_trans])
            tm = torch.tensor(self.best_times)
            ttm = torch.tensor(self.best_training_times)
            print('converge time (cold): %.3f±%.3f' % (
                tm.mean().item(), tm.std().item()))
            print('converge training time (cold): %.3f±%.3f' % (
                ttm.mean().item(), ttm.std().item()))
            print('converge epochs (cold): %.3f±%.3f' % (
                conv.mean().item(), conv.std().item()))
            if args.test_bins:
                print('score (cold):', ', '.join(
                    '%.2f±%.2f' % s
                    for s in zip(score.mean(dim=0), score.std(dim=0))))
            else:
                print('score (cold): %.2f±%.2f' % (
                    score.mean().item(), score.std().item()))

        # Output Used Time
        print('preprocessing time: %.3f' % self.preprocess_time)
        for name, times in (
            ('total training', self.training_times),
            ('total evaluation', self.evaluation_times),
        ):
            times = torch.tensor(times or [0], dtype=float)
            print('%s time: %.3f±%.3f' % (
                name, times.mean().item(), times.std().item()))

        # Output Used Space
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu = torch.cuda.max_memory_allocated(g_dev) / 1024 / 1024
        print('pre_memory: %.2fM + %.2fM = %.2fM' % (
            self.mem, self.gpu, self.mem + self.gpu))
        print('max_memory: %.2fM + %.2fM = %.2fM' % (
            mem, gpu, mem + gpu))
        print('memory_diff: %.2fM + %.2fM = %.2fM' % (
            mem - self.mem,
            gpu - self.gpu,
            mem + gpu - self.mem - self.gpu))


X, Y, E, train_masks, valid_masks, test_masks, is_bidir = load_data(
    args.dataset)
n_nodes = X.shape[0]
n_features = X.shape[1]
n_labels = int(Y.max().item() + 1)
deg = E.shape[1] / n_nodes
print('nodes: %d' % n_nodes)
print('features: %d' % n_features)
print('classes: %d' % n_labels)
print('edges without self-loops: %d' % (E.shape[1] / 2))
print('average degree: %.2f' % deg)
train_sum = sum([m.sum() for m in train_masks]) / len(train_masks)
valid_sum = sum([m.sum() for m in valid_masks]) / len(valid_masks)
if args.test_bins:
    test_sum = sum([
        sum([testbin.sum() for testbin in m])
        for m in test_masks]) / len(test_masks)
else:
    test_sum = sum([m.sum() for m in test_masks]) / len(test_masks)
print('split: %d (%.2f%%) / %d (%.2f%%) / %d (%.2f%%)' % (
    train_sum, 100 * train_sum / n_nodes,
    valid_sum, 100 * valid_sum / n_nodes,
    test_sum, 100 * test_sum / n_nodes,
))
print('intra_rate: %.2f%%' % (100 * (
    Y[E[0]] == Y[E[1]]).sum().float() / E.shape[1]))
sg = lambda x: torch.softmax(x, dim=-1)
lsg = lambda x: F.log_softmax(x, dim=-1)


def norm_adj(edges, n, sym=True):
    # Add Self-Loops
    edges = torch.cat((
        torch.arange(n).view(1, -1).repeat(2, 1), edges), dim=1)
    deg = torch.zeros(n).to(edges.device)
    deg.scatter_add_(
        dim=0, index=edges[0],
        src=torch.ones(edges.shape[1]).to(edges.device))
    # with open('degree_counts/%s_train.txt' % args.dataset, 'w') as file:
    #     for xs in deg.unique(sorted=True, return_counts=True):
    #         file.write(','.join('%d' % x for x in xs))
    #         file.write('\n')
    if sym:
        val = (deg ** -0.5)[edges].prod(dim=0)
    else:
        val = (deg ** -1)[edges[0]]
    return coo(edges, val, (n, n)).coalesce()


def eat(x, e, att1, att2, q):
    # Batch Attention from ECN
    w = torch.exp(F.leaky_relu(
        att1(x)[e] + att2(x)[e[[1, 0]]], 0.2))
    w = x.shape[0] * torch.softmax(
        w - torch.log(q).unsqueeze(0).unsqueeze(-1), dim=1)
    return w


def gat(x, e, att1, att2):
    w = torch.exp(F.leaky_relu(
        att1(x)[e[0]] + att2(x)[e[1]], 0.2))
    z = torch.zeros(x.shape[0], args.heads).to(x.device)
    z.scatter_add_(
        0, e[0].view(-1, 1).repeat(1, args.heads), w)
    return w / z[e[0]]


def mm(x, e, w, T=False):
    ax = torch.zeros(x.shape).to(x.device)
    ax.scatter_add_(
        0,
        e[-T].view(-1, 1, 1).repeat(1, x.shape[1], x.shape[2]),
        w.unsqueeze(-1) * x[e[~T]])
    return ax


def edge_sampling(edges, q, batch_size, return_weight=False):
    for perm in DataLoader(range(q.shape[0]), 2 ** 24, shuffle=True):
        p, e = q[perm], edges[:, perm]
        sampled = torch.tensor(
            list(WeightedRandomSampler(p, p.shape[0])), dtype=int)
        for cur in range(0, p.shape[0], batch_size):
            idx = sampled[cur: cur + batch_size]
            yield (e[:, idx], p[idx]) if return_weight else e[:, idx]


ev = Stat()

# Preprocessing
ev.start_preprocessing()

induc_x = x = induc_X = X = gpu(X)
induc_Y = Y = gpu(Y)
need_adj = (
    args.method == 'GraphMLP'
    or args.method == 'OKDEEM'
    or args.method == 'LPA'
    or args.method == 'ECN'
    or args.gem
    or args.correct or args.smooth
)
self_loops = gpu(torch.arange(X.shape[0]).view(1, -1).repeat(2, 1))
if need_adj:
    A = norm_adj(E, x.shape[0], args.symmetric).to(x.device)
    if args.method == 'GraphMLP' and args.batch_size:
        adj = A
        for _ in range(args.precompute - 1):
            A = torch.sparse.mm(A, adj).coalesce()
    elif args.method == 'GraphMLP':
        adj = A = A.to_dense()
        for _ in range(args.precompute - 1):
            A = A @ adj
    elif args.method == 'ECN':
        for _ in range(args.precompute):
            x = A @ x
        induc_x = x
    induc_A = A
induc_E = E = gpu(E)

ev.stop_preprocessing()

for run in range(args.runs):
    induc_mask = train_mask = train_masks[run]
    valid_mask = valid_masks[run]
    tms = test_mask = test_masks[run]
    train_y = F.one_hot(Y[train_mask], n_labels).float()
    if args.inductive:
        if args.test_bins:
            tms = torch.cat(
                test_mask).view(len(test_mask), -1).sum(dim=0).bool()
        induc_mask = train_mask[~tms]
        induc_x = induc_X = X[~tms]
        induc_Y = Y[~tms]
        induc_E = torch.cat((
            self_loops[:, ~tms],
            E[:, ~(gpu(tms)[E].sum(dim=0).bool())]
        ), dim=1).unique(return_inverse=True)[1][:, induc_X.shape[0]:]
        if need_adj:
            induc_A = norm_adj(
                induc_E.cpu(), induc_x.shape[0], args.symmetric
            ).to(induc_x.device)
            if args.method == 'GraphMLP' and args.batch_size:
                adj = induc_A
                for _ in range(args.precompute - 1):
                    induc_A = torch.sparse.mm(induc_A, adj).coalesce()
            elif args.method == 'GraphMLP':
                adj = induc_A = induc_A.to_dense()
                for _ in range(args.precompute - 1):
                    induc_A = induc_A @ adj
            elif args.method == 'ECN':
                induc_x = induc_X.clone()
                for _ in range(args.precompute):
                    induc_x = induc_A @ induc_x

    torch.manual_seed(run)
    ev.start_run()

    if args.kd:
        # https://arxiv.org/abs/2110.08727
        wocs = copy.deepcopy(args)
        wocs.runs = 0
        wocs.kd = 0.0
        kd_fn = 'predictions/%s-%s-%s.json' % (
            args.method,
            args.dataset,
            hashlib.md5(str(wocs).encode()).hexdigest())
        assert os.path.exists(kd_fn), 'no data to distill'
        with open(kd_fn) as file:
            kd = json.load(file)
        induc_y = gpu(torch.tensor(kd['run-%d' % run][1]))
        print('Load predictions made at: %s' % kd['date'])
        if args.inductive:
            induc_y = induc_y[~tms]
        net = gpu(gnn.MLP([
            n_features, args.hidden, args.hidden, n_labels
        ], dropout=args.dropout, norm='layer_norm'))
        opt = Optim([*net.parameters()])
        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                net.train()
                z = lsg(net(induc_x))
                (
                    -args.kd * (z * induc_y)[~induc_mask].sum(dim=1).mean()
                    - z[induc_mask, Y[train_mask]].mean()
                ).backward()
            ev.record_training(opt.elapsed)
            t = time.time()
            with torch.no_grad():
                net.eval()
                z = sg(net(x))
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(z):
                break
    elif args.correct or args.smooth:
        # Correct and Smooth
        # https://arxiv.org/abs/2010.13993
        wocs = copy.deepcopy(args)
        wocs.runs = 0
        wocs.correct = wocs.smooth = 0
        wocs.correct_rate = wocs.smooth_rate = 0.0
        cs_fn = 'predictions/%s-%s-%s.json' % (
            args.method,
            args.dataset,
            hashlib.md5(str(wocs).encode()).hexdigest())
        assert os.path.exists(cs_fn), 'no data to correct and smooth'
        with open(cs_fn) as file:
            cs = json.load(file)
        ev.best_acc, ev.best_y = cs['run-%d' % run]
        ev.best_y = gpu(torch.tensor(ev.best_y))
        print('Load predictions made at: %s' % cs['date'])
        opt = None
        if args.correct:
            best_acc, best_y = ev.best_acc, ev.best_y
            y = best_y
            t = time.time()
            true_err = train_y - y[train_mask]
            err = torch.zeros(y.shape).to(y.device)
            for _ in range(args.correct):
                err[train_mask] = true_err
                err = (
                    (1 - args.correct_rate) * err
                    + args.correct_rate * (A @ err))
                y = best_y + err
                acc = get_score(Y[valid_mask], y[valid_mask].argmax(dim=1))
                if acc >= best_acc:
                    best_acc, best_y = acc, y
            ev.record_training(time.time() - t)
            ev.evaluate_result(best_y)
        if args.smooth:
            best_acc, best_y = ev.best_acc, ev.best_y
            y = best_y
            t = time.time()
            for _ in range(args.smooth):
                y[train_mask] = train_y
                y = (1 - args.smooth_rate) * y + args.smooth_rate * (A @ y)
                acc = get_score(Y[valid_mask], y[valid_mask].argmax(dim=1))
                if acc >= best_acc:
                    best_acc, best_y = acc, y
            ev.record_training(time.time() - t)
            ev.evaluate_result(best_y)
    elif args.method == 'LPA':
        opt = None
        z = torch.zeros(x.shape[0], n_labels).to(x.device)
        for epoch in range(1, int(args.beta) + 1):
            t = time.time()
            z[train_mask] = train_y
            z = (1 - args.alpha) * z + args.alpha * (A @ z)
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(z):
                break
    elif args.method == 'GraphMLP' and args.batch_size:
        # https://arxiv.org/abs/2106.04051
        enc = gpu(nn.Sequential(
            nn.Linear(n_features, args.hidden),
            nn.GELU(),
            nn.LayerNorm(args.hidden),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden, args.hidden),
        ))
        pred = gpu(nn.Linear(args.hidden, n_labels))
        net = gpu(nn.Sequential(enc, pred))
        opt = Optim([*net.parameters()])
        nrange = torch.arange(induc_x.shape[0])
        induc_self_loops = gpu(nrange.view(1, -1).repeat(2, 1))
        ie, iv = induc_A.indices(), induc_A.values()
        for epoch in range(1, 1 + args.max_epochs):
            for nbatch in DataLoader(
                    nrange, batch_size=args.batch_size, shuffle=True):
                s = nbatch.shape[0]
                m = nrange < 0
                m[nbatch] = True
                em = gpu(m)[ie].prod(dim=0).bool()
                eb = torch.cat((
                    induc_self_loops[:, m], ie[:, em]
                ), dim=1).unique(return_inverse=True)[1][:, s:]
                adj = coo(eb, iv[em], (s, s)).to_dense()
                with opt:
                    net.train()
                    loss = 0
                    h = enc(induc_x[m])
                    tm = induc_mask[m]
                    if tm.any().item():
                        iy = induc_Y[induc_mask & m]
                        loss = -lsg(pred(h[tm]))[
                            torch.arange(iy.shape[0]), iy].mean()
                    h = F.normalize(h)
                    sim = h @ h.T
                    esim = torch.exp(
                        (sim - sim.diag().diag_embed()) / args.alpha)
                    loss = loss - args.beta * torch.log(
                        1e-5 + (esim * adj).sum(dim=1) / esim.sum(dim=1)
                    ).mean()
                    loss.backward()
                ev.record_training(opt.elapsed)
            t = time.time()
            with torch.no_grad():
                net.eval()
                z = sg(net(x))
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(z):
                break
    elif args.method == 'GraphMLP':
        enc = gpu(nn.Sequential(
            nn.Linear(n_features, args.hidden),
            nn.GELU(),
            nn.LayerNorm(args.hidden),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden, args.hidden),
        ))
        pred = gpu(nn.Linear(args.hidden, n_labels))
        net = gpu(nn.Sequential(enc, pred))
        opt = Optim([*net.parameters()])
        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                net.train()
                h = enc(induc_x)
                z = lsg(pred(h))
                h = F.normalize(h)
                sim = h @ h.T
                esim = torch.exp((sim - sim.diag().diag_embed()) / args.alpha)
                (
                    -z[induc_mask, Y[train_mask]].mean()
                    - args.beta * torch.log(
                        1e-5 + (esim * induc_A).sum(dim=1) / esim.sum(dim=1)
                    ).mean()
                ).backward()
            ev.record_training(opt.elapsed)
            t = time.time()
            with torch.no_grad():
                net.eval()
                z = sg(net(x))
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(z):
                break
    elif args.method == 'ECN':
        net = gpu(gnn.MLP([
            n_features, *([args.hidden] * (args.n_layers - 1)), n_labels
        ], dropout=args.dropout, norm='layer_norm'))
        opt = Optim([*net.parameters()])
        induc_e = induc_A.indices()
        em = induc_mask[induc_e[0].cpu()]
        train_e = induc_e[:, em]
        train_p = induc_A.values()[em]
        bs = args.batch_size
        if bs <= 0:
            bs = max(512, min(8 * 1024, int(
                10 + train_e.shape[1] * 0.01)))
        for epoch in range(1, 1 + args.max_epochs):
            for src, dst in edge_sampling(train_e, train_p, bs):
                net.train()
                with opt:
                    (-lsg(net(induc_x[dst]))[
                        torch.arange(dst.shape[0]), induc_Y[src]
                    ].mean()).backward()
                ev.record_training(opt.elapsed)
            t = time.time()
            with torch.no_grad():
                net.eval()
                z = A @ sg(net(x))
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(z):
                break
    elif args.method == 'OKDEEM':
        net = gpu(gnn.MLP([
            n_features,
            *([args.hidden] * (args.n_layers - 1)),
            n_labels * 2
        ], dropout=args.dropout, norm='layer_norm'))
        fwd = net.forward
        net.forward = lambda x: lsg(fwd(x).view(*x.shape[:-1], 2, n_labels))
        params = [*net.parameters()]
        opt = Optim([*net.parameters()])
        e = A.indices()
        induc_e = induc_A.indices()
        induc_p = induc_A.values()
        induc_mask = gpu(induc_mask)
        bs = args.batch_size
        if bs <= 0:
            bs = max(512, min(8 * 1024, int(
                10 + induc_e.shape[1] * 0.01)))
        for epoch in range(1, 1 + args.max_epochs):
            net.train()
            for eb, q in edge_sampling(induc_e, induc_p, bs, True):
                with opt:
                    xs, ms = induc_x[eb], induc_mask[eb]
                    zh = net(xs)
                    zh = torch.cat((zh[:, :, :1], zh[[1, 0], :, 1:]), dim=2)
                    yp = sg(zh.detach() / args.alpha)
                    loss = 0
                    if ms.any().item():
                        tys = induc_Y[eb[ms]]
                        yp[ms, :, tys] = 1
                        loss = -zh[ms, :, tys].mean()
                    loss = loss - (
                        args.gem * (
                            zh[:, :, 1] * yp[:, :, 0]).sum(dim=-1).mean()
                        + args.beta * (
                            zh[:, :, 0] * yp[:, :, 1]).sum(dim=-1).mean())
                    loss.backward()
                ev.record_training(opt.elapsed)
            t = time.time()
            with torch.no_grad():
                net.eval()
                z = sg(net(x))
                z = (A @ z[:, 1], z[:, 0])
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(z):
                break
    else:
        if args.method == 'MLP':
            net = gpu(gnn.MLP([
                n_features,
                *([args.hidden] * (args.n_layers - 1)),
                n_labels * (args.heads or 1)
            ], dropout=args.dropout, norm='layer_norm'))
            fwd = net.forward
            net.forward = lambda x, e: lsg(fwd(x))
            params = [*net.parameters()]
        else:
            in_feats = args.hidden if args.n_layers == 1 else n_features
            if args.method == 'JKNet':
                net = JKNet(in_feats, n_labels, **args.__dict__)
            elif args.method == 'GCNII':
                net = GCNII(in_feats, n_labels, **args.__dict__)
            elif args.method == 'GAT':
                net = gnn.GAT(
                    in_feats, args.hidden, args.n_layers, n_labels,
                    args.dropout, heads=args.heads)
            else:
                net = {
                    'GIN': gnn.GIN, 'GCN': gnn.GCN, 'SAGE': gnn.GraphSAGE
                }[args.method](
                    in_feats, args.hidden, args.n_layers,
                    n_labels, args.dropout)
            net = gpu(net)
            fwd = net.forward
            net.forward = lambda x, e: lsg(fwd(x, e))
            params = [*net.parameters()]
            if args.n_layers == 1:
                enc = gpu(nn.Sequential(
                    nn.Dropout(args.dropout),
                    nn.Linear(n_features, args.hidden),
                    nn.LeakyReLU()))
                params.extend([*enc.parameters()])
                net.forward = lambda x, e: lsg(fwd(enc(x), e))
        opt = Optim(params)

        if args.gem:
            if args.heads or args.batch_size:
                induc_e = induc_A.indices()
                e = A.indices()
            else:
                AT = induc_A.transpose(0, 1)
            if args.heads:
                att1 = gpu(nn.Linear(n_features, args.heads))
                att2 = gpu(nn.Linear(n_features, args.heads))
                opt.extend([*att1.parameters(), *att2.parameters()])
            if args.batch_size:
                induc_p = induc_A.values().clone()
                p = A.values()
                induc_mask = gpu(induc_mask)
                bs = args.batch_size
                if bs < 0:
                    bs = max(512, min(8 * 1024, int(
                        10 + induc_e.shape[1] * 0.01)))
                if args.beta:
                    mlp = gpu(gnn.MLP([
                        n_features, args.hidden, args.hidden, n_labels
                    ], dropout=args.dropout, norm='layer_norm'))
                    opt.extend([*mlp.parameters()])
                else:
                    deg = 2 * induc_e.shape[1] / induc_x.shape[0]
                    ema = torch.zeros((
                        induc_x.shape[0], n_labels)).to(induc_x.device)
                    ema[induc_mask] = torch.log(train_y)
        for epoch in range(1, 1 + args.max_epochs):
            if args.gem and args.batch_size:
                net.train()
                if args.beta:
                    mlp.train()
                for eb, q in edge_sampling(induc_e, induc_p, bs, True):
                    ns, iv = eb.unique(return_inverse=True, sorted=False)
                    xs = induc_x[ns]
                    with opt:
                        hs = net(xs, None)[iv[[1, 0]]]
                        if args.heads:
                            # NOTE: this is weak
                            hs = (
                                hs.view(2, -1, args.heads, n_labels)
                                * eat(xs, iv, att1, att2, q).unsqueeze(-1))
                        # TODO: overwrite pseudo labels
                        if args.beta:
                            gs = lsg(mlp(xs))[iv]
                            if args.heads:
                                gs = gs.unsqueeze(2)
                            loss = -args.beta * (
                                gs * sg(hs.detach() / args.alpha)
                            ).sum(dim=-1).mean()
                            zs = sg(gs.detach() / args.alpha)
                        else:
                            loss = 0
                            zs = sg(ema[eb])
                            if args.heads:
                                zs = zs.unsqueeze(2)
                        loss = loss - args.gem * (hs * zs).sum(dim=-1).mean()
                        ms = induc_mask[eb]
                        if ms.any().item():
                            tys = induc_Y[eb[ms]]
                            if args.heads:
                                loss = loss - hs[ms, :, tys].mean()
                                if args.beta:
                                    loss = loss - gs[ms, :, tys].mean()
                            else:
                                loss = loss - hs[ms, tys].mean()
                                if args.beta:
                                    loss = loss - gs[ms, tys].mean()
                        loss.backward()
                        if not args.beta:
                            if args.heads:
                                hs = hs.mean(dim=2)
                            with torch.no_grad():
                                ema.scatter_add_(
                                    0,
                                    eb.unsqueeze(-1).repeat(
                                        1, 1, n_labels).view(-1, n_labels),
                                    hs.view(-1, n_labels) / deg)
                    ev.record_training(opt.elapsed)
                if not args.beta:
                    # NOTE: 1 + x + xx + xxx + ... ~ 1 / (1 - x)
                    ema *= 1 - args.alpha
            else:
                with opt:
                    net.train()
                    z = net(induc_x, induc_E)
                    if args.gem:
                        if args.heads:
                            w = gat(induc_x, induc_e, att1, att2)
                            z = z.view(-1, args.heads, n_labels)
                            az = lsg(mm(z, induc_e, w))
                        else:
                            az = lsg(induc_A @ z)
                        with torch.no_grad():
                            y = sg(az / args.alpha)
                            y[induc_mask] = 0
                            if args.heads:
                                y[induc_mask, :, Y[train_mask]] = 1
                                ay = mm(y, induc_e, w, True)
                            else:
                                y[induc_mask, Y[train_mask]] = 1
                                ay = AT @ y
                        if args.heads:
                            loss = -az[induc_mask, :, Y[train_mask]].mean()
                        else:
                            loss = -az[induc_mask, Y[train_mask]].mean()
                        loss = loss - (z * ay).sum(dim=-1).mean() * args.gem
                    elif args.er:
                        y = sg(z / args.alpha)
                        loss = (
                            -args.er * (z * y).sum(dim=1).mean()
                            - z[induc_mask, Y[train_mask]].mean())
                    else:
                        loss = -lsg(z)[induc_mask, Y[train_mask]].mean()
                    loss.backward()
                ev.record_training(opt.elapsed)

            t = time.time()
            with torch.no_grad():
                net.eval()
                z = net(x, E)
                if args.gem:
                    if args.heads:
                        w = gat(x, e, att1, att2)
                        z = sg(z.view(-1, args.heads, n_labels))
                        z = F.normalize(mm(z, e, w).mean(dim=1), p=1)
                    else:
                        z = F.normalize(A @ sg(z), p=1)
                    if args.batch_size and args.beta:
                        mlp.eval()
                        z = (z, sg(mlp(x)))
                else:
                    z = sg(z)
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(z):
                break

    ev.end_run()
ev.end_all()
print('params: 0' if opt is None else opt)
print('script time:', time.time() - script_time)
