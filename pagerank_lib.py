import numpy as np, scipy.sparse as sp, time, networkx as nx

def load_clean(path="web_synth.txt", nmax=None):
    E = np.loadtxt(path, dtype=np.int64, comments="#")
    raw = len(E)
    if nmax is not None:
        E = E[(E[:,0] < nmax) & (E[:,1] < nmax)]
    n_in = len(E)
    E = E[E[:,0] != E[:,1]]
    n_noself = len(E)
    E = np.unique(E, axis=0)
    N = nmax if nmax is not None else int(E.max()) + 1
    return E, N, dict(raw=raw, in_subset=n_in, self_loops=n_in-n_noself, dups=n_noself-len(E), clean=len(E))

def build_matrix(E, N):
    outdeg = np.bincount(E[:,0], minlength=N).astype(float)
    w = 1.0 / outdeg[E[:,0]]
    # column-stochastic transition: M[j,i] = 1/L(i) for edge i->j
    M = sp.csr_matrix((w, (E[:,1], E[:,0])), shape=(N, N))
    return M, outdeg

def pagerank_numpy(E, N, d=0.85, tol=1e-6, max_iter=200):
    M, outdeg = build_matrix(E, N)
    dangling = outdeg == 0
    r = np.full(N, 1.0/N)
    errs = []
    for it in range(1, max_iter+1):
        dm = r[dangling].sum()
        rn = d*(M @ r + dm/N) + (1-d)/N
        err = np.abs(rn - r).sum()
        errs.append(err); r = rn
        if err < tol: break
    return r, it, errs

def pagerank_nx(E, N, d=0.85, tol=1e-6):
    G = nx.DiGraph(); G.add_nodes_from(range(N)); G.add_edges_from(map(tuple, E.tolist()))
    pr = nx.pagerank(G, alpha=d, tol=tol/N, max_iter=200)
    return np.array([pr[i] for i in range(N)])

def indegree(E, N):
    return np.bincount(E[:,1], minlength=N)
