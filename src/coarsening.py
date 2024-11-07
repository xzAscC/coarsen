import numpy as np
import pygsp as gsp
from pygsp import graphs, filters, reduction
import scipy as sp
from scipy import sparse
from sortedcontainers import SortedList
import time
import utils
import maxWeightMatching

def coarsen(
    G,
    K=10,
    r=0.5,
    max_levels=10,
    method="lam",
    algorithm="greedy",
    Uk=None,
    lk=None,
    max_level_r=0.99,
):
    r = np.clip(r, 0, 0.999)
    G0 = G

    N = G.N

    # current and target graph sizes
    n, n_target = N, np.ceil((1 - r) * N)

    C = sp.sparse.eye(N, format="csc")
    Gc = G

    Call, Gall = [], []
    Gall.append(G)

    for level in range(1, max_levels + 1):

        begin_time = time.time()
        G = Gc

        # how much more we need to reduce the current graph
        r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

        if level == 1:
            if (Uk is not None) and (lk is not None) and (len(lk) >= K):
                mask = lk < 1e-10
                lk[mask] = 1
                lsinv = lk ** (-0.5)
                lsinv[mask] = 0
                B = Uk[:, :K] @ np.diag(lsinv[:K])
            else:
                offset = 2 * max(G.dw)
                T = offset * sp.sparse.eye(G.N, format="csc") - G.L
                lk, Uk = sp.sparse.linalg.eigsh(T, k=K, which="LM", tol=1e-5)
                lk = (offset - lk)[::-1]
                Uk = Uk[:, ::-1]
                mask = lk < 1e-10
                lk[mask] = 1
                lsinv = lk ** (-0.5)
                lsinv[mask] = 0
                B = Uk @ np.diag(lsinv)
            A = B
        else:
            B = iC.dot(B)
            d, V = np.linalg.eig(B.T @ (G.L).dot(B))
            mask = d == 0
            d[mask] = 1
            dinvsqrt = d ** (-1 / 2)
            dinvsqrt[mask] = 0
            A = B @ np.diag(dinvsqrt) @ V

        coarsening_list = contract_variation_edges(
            G, K=K, A=A, r=r_cur, algorithm=algorithm
        )



        iC = get_coarsening_matrix(G, coarsening_list)

        if iC.shape[1] - iC.shape[0] <= 2:
            break  # avoid too many levels for so few nodes

        C = iC.dot(C)
        Call.append(iC)

        Wc = utils.zero_diag(
            coarsen_matrix(G.W, iC)
        )  # coarsen and remove self-loops
        Wc = (
            Wc + Wc.T
        ) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc = gsp.graphs.Graph(Wc)
        else:
            Gc = gsp.graphs.Graph(Wc, coords=(iC.power(2)).dot(G.coords))
        Gall.append(Gc)

        n = Gc.N
        end_time = time.time()
        print("Level %d: %d nodes, %.2f sec" % (level, n, end_time - begin_time))

        if n <= n_target:
            break

    return GC


def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))


def contract_variation_edges(G, A=None, K=10, r=0.5, algorithm="greedy"):
    N, deg, M = G.N, G.dw, G.Ne
    ones = np.ones(2)
    Pibot = np.eye(2) - np.outer(ones, ones) / 2

    def subgraph_cost(G, A, edge):
        edge, w = edge[:2].astype(np.int_), edge[2]
        deg_new = 2 * deg[edge] - w
        L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return np.linalg.norm(B.T @ L @ B)

    def subgraph_cost_old(G, A, edge):
        w = G.W[edge[0], edge[1]]
        deg_new = 2 * deg[edge] - w
        L = np.array([[deg_new[0], -w], [-w, deg_new[1]]])
        B = Pibot @ A[edge, :]
        return np.linalg.norm(B.T @ L @ B)

    edges = np.array(G.get_edge_list())
    weights = np.array([subgraph_cost(G, A, edges[:, e]) for e in range(M)])


    coarsening_list = matching_optimal(G, weights=weights, r=r)

    return coarsening_list

def matching_optimal(G, weights, r=0.9):

    N = G.N

    edges = G.get_edge_list()
    edges = np.array(edges[0:2])
    M = edges.shape[1]

    max_weight = 1 * np.max(weights)

    edge_list = []
    for edgeIdx in range(M):
        [i, j] = edges[:, edgeIdx]
        if i == j:
            continue
        edge_list.append((i, j, max_weight - weights[edgeIdx]))

    assert min(weights) >= 0

    tmp = np.array(maxWeightMatching.maxWeightMatching(edge_list))

    m = tmp.shape[0]
    matching = np.zeros((m, 2), dtype=int)
    matching[:, 0] = range(m)
    matching[:, 1] = tmp

    idx = np.where(tmp != -1)[0]
    matching = matching[idx, :]
    idx = np.where(matching[:, 0] > matching[:, 1])[0]
    matching = matching[idx, :]

    assert matching.shape[0] >= 1

    matched_weights = np.zeros(matching.shape[0])
    for mIdx in range(matching.shape[0]):
        i = matching[mIdx, 0]
        j = matching[mIdx, 1]
        eIdx = [
            e
            for e, t in enumerate(edges[:, :].T)
            if ((t == [i, j]).all() or (t == [j, i]).all())
        ]
        matched_weights[mIdx] = weights[eIdx]

    keep = min(int(np.ceil(r * N)), matching.shape[0])
    if keep < matching.shape[0]:
        idx = np.argpartition(matched_weights, keep)
        idx = idx[0:keep]
        matching = matching[idx, :]

    return matching


def get_coarsening_matrix(G, partitioning):

    # C = np.eye(G.N)
    C = sp.sparse.eye(G.N, format="lil")

    rows_to_delete = []
    for subgraph in partitioning:

        nc = len(subgraph)

        C[subgraph[0], subgraph] = 1 / np.sqrt(nc)  

        rows_to_delete.extend(subgraph[1:])



    C.rows = np.delete(C.rows, rows_to_delete)
    C.data = np.delete(C.data, rows_to_delete)
    C._shape = (G.N - len(rows_to_delete), G.N)

    C = sp.sparse.csc_matrix(C)


    return C
