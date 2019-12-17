import numpy as np
import scipy.sparse as sp
from GF_Attack import utils
from scipy import linalg

class GFA:

    def __init__(self, adj, z_obs, u, X_mean, eig_vals, eig_vec, K, T, perturb_logs):

        # Adjacency matrix
        self.adj = adj.copy().tolil()
        self.adj_no_selfloops = self.adj.copy()
        self.adj_no_selfloops.setdiag(0)
        self.adj_orig = self.adj.copy().tolil()
        self.u = u  # the node being attacked
        self.adj_preprocessed = utils.preprocess_graph(self.adj).tolil()
        # Number of nodes
        self.N = adj.shape[0]

        # Node attributes
        self.X_mean = X_mean

        self.eig_vals = eig_vals
        self.eig_vec = eig_vec

        #The order of graph filter K
        self.K = K

        #Top-T largest singular values/vectors selected
        self.T = T

        # Node labels
        self.z_obs = z_obs.copy()
        self.label_u = self.z_obs[self.u]
        self.K = np.max(self.z_obs)+1

        self.structure_perturbations = []

        self.potential_edges = []
        self.save_perturb_logs = perturb_logs


    def attack_model(self, n_perturbations, delta_cutoff=0.004):
        """
        Perform an attack on the surrogate model.

        Parameters
        ----------
        n_perturbations: int
            The number of perturbations (structure or feature) to perform.
        delta_cutoff: float
            The critical value for the likelihood ratio test of the power law distributions.
             See the Chi square distribution with one degree of freedom. Default value 0.004
             corresponds to a p-value of roughly 0.95.

        Returns
        -------
        None.

        """

        assert n_perturbations > 0, "need at least one perturbation"

        print("##### Starting attack #####")
        print("##### Attacking structure #####")
        print("##### Performing {} perturbations #####".format(n_perturbations))


        # Setup starting values of the likelihood ratio test.
        degree_sequence_start = self.adj_orig.sum(0).A1
        current_degree_sequence = self.adj.sum(0).A1
        d_min = 2 #denotes the minimum degree a node needs to have to be considered in the power-law test
        S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
        current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
        n_start = np.sum(degree_sequence_start >= d_min)
        current_n = np.sum(current_degree_sequence >= d_min)
        alpha_start = compute_alpha(n_start, S_d_start, d_min)
        log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

        # direct attack
        self.potential_edges = np.column_stack((np.tile(self.u, self.N-1), np.setdiff1d(np.arange(self.N), self.u)))

        self.potential_edges = self.potential_edges.astype("int32")
        for _ in range(n_perturbations):
            print("##### ...{}/{} perturbations ... #####".format(_+1, n_perturbations))
            # Do not consider edges that, if removed, result in singleton edges in the graph.
            singleton_filter = filter_singletons(self.potential_edges, self.adj)
            filtered_edges = self.potential_edges[singleton_filter]

            # Update the values for the power law likelihood ratio test.
            deltas = 2 * (1 - self.adj[tuple(filtered_edges.T)].toarray()[0] )- 1
            d_edges_old = current_degree_sequence[filtered_edges]
            d_edges_new = current_degree_sequence[filtered_edges] + deltas[:, None]
            new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
            new_alphas = compute_alpha(new_n, new_S_d, d_min)
            new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
            alphas_combined = compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
            new_ll_combined = compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
            new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)

            # Do not consider edges that, if added/removed, would lead to a violation of the
            # likelihood ration Chi_square cutoff value.
            powerlaw_filter = filter_chisquare(new_ratios, delta_cutoff)
            filtered_edges_final = filtered_edges[powerlaw_filter]


            # Compute the struct scores for each potential edge as described in paper.
            struct_scores = utils.cal_scores(self.adj, self.X_mean, self.eig_vals, self.eig_vec, filtered_edges_final, K=self.K, T=self.T, lambda_method = "nosum")

            struct_scores = struct_scores.reshape(struct_scores.shape[0],1)

            best_edge_ix = struct_scores.argmax()
            best_edge_score = struct_scores.max()
            best_edge = filtered_edges_final[best_edge_ix]
            while (tuple(best_edge) in self.structure_perturbations):
                struct_scores[best_edge_ix] = 0
                best_edge_ix = struct_scores.argmax()
                best_edge_score = struct_scores.max()
                best_edge = filtered_edges_final[best_edge_ix]

            label_ori = self.adj[tuple(best_edge)]
            self.adj[tuple(best_edge)] = self.adj[tuple(best_edge[::-1])] = 1 - self.adj[tuple(best_edge)]

            with open(self.save_perturb_logs,"a+") as f:
                f.write(str(best_edge) + ' ' + str(label_ori) + '\n')

            self.adj_preprocessed = utils.preprocess_graph(self.adj)

            self.structure_perturbations.append(tuple(best_edge))

            # Update likelihood ratio test values
            current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
            current_n = new_n[powerlaw_filter][best_edge_ix]
            current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]

    def reset(self):
        """
        Reset GFA
        """
        self.adj = self.adj_orig.copy()
        self.structure_perturbations = []
        self.potential_edges = []


def compute_alpha(n, S_d, d_min):
    """
    Approximate the alpha of a power law distribution.

    Parameters
    ----------
    n: int or np.array of int
        Number of entries that are larger than or equal to d_min

    S_d: float or np.array of float
         Sum of log degrees in the distribution that are larger than or equal to d_min

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    alpha: float
        The estimated alpha of the power law distribution
    """

    return n / (S_d - n * np.log(d_min - 0.5)) + 1


def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.

    Parameters
    ----------
    S_old: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.

    n_old: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    d_old: np.array, shape [N,] dtype int
        The old degree sequence.

    d_new: np.array, shape [N,] dtype int
        The new degree sequence

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    new_S_d: float, the updated sum of log degrees in the distribution that are larger than or equal to d_min.
    new_n: int, the updated number of entries in the old distribution that are larger than or equal to d_min.
    """

    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)

    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

    return new_S_d, new_n


def compute_log_likelihood(n, alpha, S_d, d_min):
    """
    Compute log likelihood of the powerlaw fit.

    Parameters
    ----------
    n: int
        Number of entries in the old distribution that are larger than or equal to d_min.

    alpha: float
        The estimated alpha of the power law distribution

    S_d: float
         Sum of log degrees in the distribution that are larger than or equal to d_min.

    d_min: int
        The minimum degree of nodes to consider

    Returns
    -------
    float: the estimated log likelihood
    """

    return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d

def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.

    Parameters
    ----------
    edges: np.array, shape [P, 2], dtype int, where P is the number of input edges.
        The potential edges.

    adj: sp.sparse_matrix, shape [N,N]
        The input adjacency matrix.

    Returns
    -------
    np.array, shape [P, 2], dtype bool:
        A binary vector of length len(edges), False values indicate that the edge at
        the index  generates singleton edges, and should thus be avoided.

    """

    degs = np.squeeze(np.array(np.sum(adj,0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1

    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0

def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff
