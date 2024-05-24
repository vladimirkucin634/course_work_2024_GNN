import dgl.function as fn
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from const import TRAIN_SIZE
import dgl
from itertools import combinations_with_replacement, permutations
from typing import Callable
import re
import tqdm

class DotPredictor(torch.nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]

class EdgePred(torch.nn.Module):
    def forward(self, g, h, op='cos'):
        src, dst = g.edges()
        h_src = h[src]
        h_dst = h[dst]
        return dgl.nn.EdgePredictor(op=op)(h_src, h_dst)[:, 0]
    
class MLPPredictor(torch.nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = torch.nn.Linear(h_feats * 2, h_feats)
        self.W2 = torch.nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(torch.nn.functional.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def compute_auc(pos_score: torch.Tensor, neg_score: torch.Tensor) -> float:
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> float:
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)

def train(
        model: torch.nn.Module, 
        forward_args: list,
        epoch: int, 
        train_pos_g: dgl.DGLGraph, 
        train_neg_g: dgl.DGLGraph, 
        test_pos_g: dgl.DGLGraph, 
        test_neg_g: dgl.DGLGraph, 
        optimizer: torch.optim, 
        pred, 
        save_logs: bool = True, 
        print_results: bool = False, 
        verbose=20, # save logs every 'verbose' epochs
        save_model: bool = False # path to save model
        ):
    
    losses, aucs = [], []

    for e in tqdm(range(epoch)):
        # forward
        h = model(*forward_args)        
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % verbose == 0:
            with torch.no_grad():
                pos_score = pred(test_pos_g, h)
                neg_score = pred(test_neg_g, h)
                roc_auc = compute_auc(pos_score, neg_score)
                if save_logs:
                    losses.append(loss.item())
                    aucs.append(roc_auc)
                if print_results:
                    print('In epoch {}, loss: {}, test auc: {}'.format(e, loss.item(), roc_auc))
        
    if save_model:
        torch.save(model, f'{save_model}.pth')
        print(f'Model saved at {save_model}.pth')

    if save_logs:
        return model, losses, aucs
    
    return model

def get_combs(vars: list | tuple, c: int) -> list:
    return list(
        set(
            permutations(vars, c)) 
        | 
        set(
            combinations_with_replacement(vars, c)
            )
        )

def get_train_test_graphs(g: dgl.DGLGraph, train_s: float = TRAIN_SIZE) -> tuple:

    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)

    train_size = int(len(eids) * train_s)
    test_size = g.number_of_edges() - train_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    adj = g.adj_external(scipy_fmt='coo')
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())

    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)

    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])
    test_g = dgl.remove_edges(g, eids[test_size:])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())

    return train_g, test_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

def preproc_strigs(string: str) -> str:
    return re.sub(r'[^а-яА-Я\s]', '', string).strip()