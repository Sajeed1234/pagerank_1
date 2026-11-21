import numpy as np, time, json, sys
from scipy.stats import spearmanr
from pagerank_lib import *

res = {}
E, N, pre = load_clean()
res["pre"] = pre; res["N"] = N
outdeg = np.bincount(E[:,0], minlength=N)
res["dangling"] = int((outdeg==0).sum())
print(pre, N, res["dangling"])

# ---- 1. NumPy/SciPy
t=time.time(); r_np, it_np, errs = pagerank_numpy(E, N); t_np=time.time()-t
res["np"] = dict(time=t_np, iters=it_np, errs=errs, sum=float(r_np.sum()))
print("numpy", t_np, it_np, r_np.sum())

# ---- 2. NetworkX
t=time.time(); r_nx = pagerank_nx(E, N); t_nx=time.time()-t
res["nx"] = dict(time=t_nx)
print("nx", t_nx)
res["np_vs_nx"] = dict(max_abs_diff=float(np.abs(r_np-r_nx).max()),
                       l1=float(np.abs(r_np-r_nx).sum()),
                       top20_overlap=len(set(np.argsort(-r_np)[:20]) & set(np.argsort(-r_nx)[:20])),
                       spearman=float(spearmanr(r_np, r_nx)[0]))
print(res["np_vs_nx"])

# ---- 3. in-degree comparison
indeg = indegree(E, N)
top_pr = np.argsort(-r_np)[:20]; top_in = np.argsort(-indeg, kind="stable")[:20]
res["indeg"] = dict(top20_overlap=len(set(top_pr)&set(top_in)),
                    top10_overlap=len(set(np.argsort(-r_np)[:10])&set(np.argsort(-indeg,kind="stable")[:10])),
                    spearman=float(spearmanr(r_np, indeg)[0]))
# nodes with same in-degree but very different PageRank: show a concrete example
# pick nodes in top-100 by PR with lowest in-degree rank
order_pr = np.argsort(-r_np); rank_pr = np.empty(N,int); rank_pr[order_pr]=np.arange(N)
order_in = np.argsort(-indeg, kind="stable"); rank_in = np.empty(N,int); rank_in[order_in]=np.arange(N)
diff = rank_in - rank_pr   # large positive => PageRank ranks it much higher than in-degree
cand = [i for i in order_pr[:200]]
cand.sort(key=lambda i: -(rank_in[i]-rank_pr[i]))
res["indeg"]["biggest_movers"] = [dict(node=int(i), pr_rank=int(rank_pr[i]+1), indeg_rank=int(rank_in[i]+1),
                                   indeg=int(indeg[i]), pr=float(r_np[i])) for i in cand[:5]]
res["top20"] = [dict(rank=k+1, node=int(i), pr=float(r_np[i]), indeg=int(indeg[i]), outdeg=int(outdeg[i])) for k,i in enumerate(top_pr)]
print(res["indeg"])

# ---- 4. damping sensitivity
base10 = set(np.argsort(-r_np)[:10]); base20=set(np.argsort(-r_np)[:20])
ds = []
for d in [0.5,0.7,0.85,0.95]:
    r, it, e = pagerank_numpy(E, N, d=d)
    ds.append(dict(d=d, iters=it, top10_overlap=len(set(np.argsort(-r)[:10])&base10),
                   top20_overlap=len(set(np.argsort(-r)[:20])&base20),
                   top1=int(np.argmax(r)), top1_pr=float(r.max()), spearman=float(spearmanr(r, r_np)[0]), errs=e))
    print(ds[-1]["d"], it, ds[-1]["top10_overlap"], ds[-1]["top1_pr"])
res["damping"] = ds

# ---- 5. power-law fit on PR distribution (log-binned)
vals = r_np[r_np>0]
bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 30)
h,_ = np.histogram(vals, bins=bins, density=True)
c = np.sqrt(bins[1:]*bins[:-1]); m = h>0
# fit tail only: above median PR
tail = m & (c > np.median(vals)*3)
slope = np.polyfit(np.log10(c[tail]), np.log10(h[tail]), 1)[0]
res["powerlaw_slope"] = float(slope)
res["pr_stats"] = dict(min=float(vals.min()), median=float(np.median(vals)), max=float(vals.max()), uniform=1.0/N)
print("slope", slope)

np.save("r_np.npy", r_np); np.save("indeg.npy", indeg)
json.dump(res, open("results_core.json","w"), indent=1)
