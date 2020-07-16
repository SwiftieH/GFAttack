# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy import linalg
import math
from numba import jit


def load_npz_edges(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    dict_of_lists = {}
    with np.load(file_name) as loader:
        loader = dict(loader)
        num_nodes = loader['adj_shape'][0]
        indices = loader['adj_indices']
        indptr = loader['adj_indptr']
        for i in range(num_nodes):
            if len(indices[indptr[i]:indptr[i+1]]) > 0:
                dict_of_lists[i] = indices[indptr[i]:indptr[i+1]].tolist()

    return dict_of_lists


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components ( adj ) #Return the length-N array of each node's label in the connected components.
    component_sizes = np.bincount(component_indices) #Count number of occurrences of each value in array of non-negative ints.
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized

def cal_scores(A, X_mean, eig_vals, eig_vec, filtered_edges, K=2, T=128, lambda_method = "nosum"):
    '''
    Calculate the scores as formulated in paper.

    Parameters
    ----------
    K: int, default: 2
        The order of graph filter K.

    T: int, default: 128
        Selecting the Top-T smallest eigen-values/vectors.

    lambda_method: "sum"/"nosum", default: "nosum"
        Indicates the scores are calculated from which loss as in Equation (8) or Equation (12).
        "nosum" denotes Equation (8), where the loss is derived from Graph Convolutional Networks,
        "sum" denotes Equation (12), where the loss is derived from Sampling-based Graph Embedding Methods.

    Returns
    -------
    Scores for candidate edges.

    '''
    results = []
    A = A + sp.eye(A.shape[0])
    A[A > 1] = 1
    rowsum = A.sum(1).A1
    D_min = rowsum.min()
    abs_V = len(eig_vals)
    tmp_k = T

    return_values = []

    for j in range(len(filtered_edges)):
        filtered_edge = filtered_edges[j]
        eig_vals_res = np.zeros(len(eig_vals))
        eig_vals_res = (1 - 2*A[filtered_edge[0], filtered_edge[1]]) * (2* eig_vec[filtered_edge[0],:] * eig_vec[filtered_edge[1],:] - eig_vals *
                                                                        ( np.square(eig_vec[filtered_edge[0],:]) + np.square(eig_vec[filtered_edge[1],:])))
        eig_vals_res = eig_vals + eig_vals_res

        if lambda_method == "sum":
            if K==1:
                eig_vals_res =np.abs(eig_vals_res / K) * (1/D_min)
            else:
                for itr in range(1,K):
                    eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr+1)
                eig_vals_res = np.abs(eig_vals_res / K) * (1/D_min)
        else:
            eig_vals_res = np.square((eig_vals_res + np.ones(len(eig_vals_res))))
            eig_vals_res = np.power(eig_vals_res, K)

        eig_vals_idx = np.argsort(eig_vals_res)  # from small to large
        eig_vals_k_sum = eig_vals_res[eig_vals_idx[:tmp_k]].sum()
        u_k = eig_vec[:,eig_vals_idx[:tmp_k]]
        u_x_mean = u_k.T.dot(X_mean)
        return_values.append(eig_vals_k_sum * np.square(np.linalg.norm(u_x_mean)))

        print("The one_edge_version progress:%f%%" % (((j + 1) / (len(filtered_edges))) * 100), end='\r', flush=True)
    print("\n")

    return np.array(return_values)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features



