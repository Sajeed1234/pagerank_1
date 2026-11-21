import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
core = json.load(open("results_core.json")); scale = json.load(open("results_scale.json"))
r = np.load("r_np.npy"); indeg = np.load("indeg.npy")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":GRID,"axes.labelcolor":INK2,
    "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"figure.facecolor":"white","axes.facecolor":"white"})
def save(fig, name): fig.tight_layout(); fig.savefig(f"fig/{name}.png", dpi=200); plt.close(fig)

# 1 top 20
t = core["top20"][::-1]
fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.barh([f"Page {x['node']}" for x in t], [x["pr"] for x in t], color=BLUE, height=0.62)
for i, x in enumerate(t): ax.text(x["pr"]+0.0004, i, f"{x['pr']:.4f}", va="center", fontsize=8, color=INK2)
ax.set_xlabel("PageRank score (d = 0.85)"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(x["pr"] for x in t)*1.15); ax.set_title("Top 20 pages by PageRank", loc="left", color=INK, fontsize=11)
save(fig, "top20")

# 2 rank-size
o = np.sort(r)[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.loglog(np.arange(1, len(o)+1), o, color=BLUE, lw=1.6)
ax.set_xlabel("Rank of page (log scale)"); ax.set_ylabel("PageRank score (log scale)")
ax.set_title("PageRank rank-size distribution", loc="left", color=INK, fontsize=11)
save(fig, "loglog")

# 3 convergence for d
fig, ax = plt.subplots(figsize=(6.4, 4.0))
cols = {0.5:BLUE, 0.7:ORANGE, 0.85:AQUA, 0.95:YELLOW}
for d in core["damping"]:
    e = d["errs"]; x = np.arange(1, len(e)+1)
    ax.semilogy(x, e, color=cols[d["d"]], lw=1.8, label=f"d = {d['d']}")
    ax.text(x[-1]+0.3, e[-1], f"d={d['d']}", color=INK2, fontsize=8, va="center")
ax.axhline(1e-6, color=INK2, lw=0.8, ls="--"); ax.text(1, 1.4e-6, "tolerance 1e-6", color=INK2, fontsize=8)
ax.set_xlim(0, 26.5); ax.set_xlabel("Iteration"); ax.set_ylabel("L1 change between iterations")
ax.legend(frameon=False, fontsize=8); ax.set_title("Convergence for different damping factors", loc="left", color=INK, fontsize=11)
save(fig, "convergence")

# 4 runtime scaling
n = [s["n"] for s in scale]
tot_np = [s["numpy"]["build"]+s["numpy"]["iterate"] for s in scale]
tot_nx = [s["networkx"]["build"]+s["networkx"]["iterate"] for s in scale]
tot_sp = [s["spark"]["prep"]+s["spark"]["iterate"] for s in scale]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for y, c, lab in [(tot_np, BLUE, "NumPy/SciPy"), (tot_nx, ORANGE, "NetworkX"), (tot_sp, AQUA, "PySpark (local[2])")]:
    ax.semilogy(n, y, color=c, lw=1.8, marker="o", ms=5, mec="white", mew=1, label=lab)
    ax.text(n[-1]+1500, y[-1], lab, color=INK2, fontsize=8, va="center")
ax.set_xlim(5000, 128000); ax.set_xticks(n); ax.set_xticklabels([f"{v//1000}k" for v in n])
ax.set_xlabel("Number of pages (nodes)"); ax.set_ylabel("Total time in seconds (log scale)")
ax.set_title("Runtime versus graph size", loc="left", color=INK, fontsize=11)
save(fig, "runtime")

# 5 PageRank vs in-degree rank
order_pr = np.argsort(-r); rank_pr = np.empty(len(r), int); rank_pr[order_pr] = np.arange(1, len(r)+1)
order_in = np.argsort(-indeg, kind="stable"); rank_in = np.empty(len(r), int); rank_in[order_in] = np.arange(1, len(r)+1)
idx = order_pr[:2000]
fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.scatter(rank_in[idx], rank_pr[idx], s=5, color=BLUE, alpha=0.55, linewidths=0)
lim = [1, 2500]; ax.plot(lim, lim, color=INK2, lw=0.8, ls="--"); ax.text(60, 30, "same rank", color=INK2, fontsize=8, rotation=0)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 2500); ax.set_ylim(1, 2500)
ax.set_xlabel("Rank by in-degree"); ax.set_ylabel("Rank by PageRank")
ax.set_title("PageRank rank vs in-degree rank (top 2,000 pages)", loc="left", color=INK, fontsize=11)
save(fig, "indegree")
print("ok")
