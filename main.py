#!/usr/bin/env python
# coding: utf-8

import time
from matplotlib import pyplot as plt

from GF_Attack import utils, GCN
from GF_Attack import GFA
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os
import scipy.sparse as sp
gpu_id = 0
import scipy.sparse as sp
from scipy import linalg

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


cur_path = os.path.abspath('.')
for path in ["citeseer","cora","pubmed"]:
    if not os.path.exists(os.path.join(cur_path, 'GF_Attack_logs', path)):
            os.makedirs(os.path.join(cur_path, 'GF_Attack_logs', path))

parser = ArgumentParser("rdlink_gcn",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("--dataset", required=True, help='dataset string.') # 'citeseer', 'cora', 'pubmed'
args = parser.parse_args()


dataset = args.dataset
_A_obs = sp.load_npz("data/" + dataset + "_adj.npz")
_X_obs = sp.load_npz("data/" + dataset + "_features.npz")
_z_obs = np.load("data/" + dataset + "_labels.npy")

perturb_save_logs = os.path.join(cur_path, 'GF_Attack_logs/' + dataset + '/Pert_Edge_lists_bestone.txt')

_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1
lcc = utils.largest_connected_components(_A_obs)

_A_obs = _A_obs[lcc][:,lcc] #Use the largest connected_component for train

assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes" #each node should have at least one edge

_X_obs = sp.csr_matrix(_X_obs[lcc].astype('float32'))
_z_obs = _z_obs[lcc]
_X_obs = _X_obs.astype('float32')
_N = _A_obs.shape[0]
_K = _z_obs.max()+1
_Z_obs = np.eye(_K)[_z_obs]
_An = utils.preprocess_graph(_A_obs)

sizes = [16, _K]
degrees = _A_obs.sum(0).A1

X_mean = np.sum(_X_obs, axis = 1)
K = 2
T = _N//2 #Default as 128

seed = 15
unlabeled_share = 0.8
val_share = 0.1
train_share = 1 - unlabeled_share - val_share
np.random.seed(seed)

split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(np.arange(_N),
                                                                       train_size=train_share,
                                                                       val_size=val_share,
                                                                       test_size=unlabeled_share,
                                                                       stratify=_z_obs)


gcn_before = GCN.GCN(sizes, _An, utils.preprocess_features(_X_obs), "gcn", gpu_id=gpu_id)

gcn_before.train(_An, split_train, split_val, _Z_obs)
gcn_before.test ( split_unlabeled, _Z_obs, _An)


pbar = tqdm(range(len(split_unlabeled)))
attacked = 0

A_processed = _An
A_I = _A_obs + sp.eye(_A_obs.shape[0])
A_I[A_I > 1] = 1
rowsum = A_I.sum(1).A1

degree_mat = sp.diags(rowsum)

eig_vals, eig_vec = linalg.eigh(A_I.todense(), degree_mat.todense())


for pos in pbar:
    u = split_unlabeled[pos]
    print ('Before perturbation:')
    if gcn_before.test ([u], _Z_obs, _An ) == 0:
        attacked += 1
        continue

    n_perturbations = 1 #single edge perturbation.

    start = time.time()

    GF_Attack = GFA.GFA(_A_obs, _z_obs, u, X_mean, eig_vals, eig_vec, K, T, perturb_save_logs)
    end = time.time()
    print ('GF_Attack time:'+str(end - start))
    GF_Attack.reset()
    GF_Attack.attack_model(n_perturbations)

    print(GF_Attack.structure_perturbations)

    print ('After perturbation:')
    acc_after = gcn_before.test ( [u], _Z_obs, GF_Attack.adj_preprocessed)
    if acc_after < 1.0:
        attacked += 1

    pbar.set_description('current attack: {}'.format(attacked))

print('Final accuracy after attack is: {} \n'.format(1.0 - attacked/len(split_unlabeled)))
gcn_before.test ( split_unlabeled, _Z_obs , _An)
