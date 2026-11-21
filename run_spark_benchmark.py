import numpy as np, time, json, os, sys
from operator import add
from scipy.stats import spearmanr
import scipy.sparse as sp
from pagerank_lib import *
import networkx as nx
from pyspark.sql import SparkSession

spark = (SparkSession.builder.master("local[2]").appName("PageRank")
         .config("spark.default.parallelism","4").config("spark.ui.enabled","false")
         .config("spark.driver.memory","3g").config("spark.sql.shuffle.partitions","4").getOrCreate())
sc = spark.sparkContext; sc.setLogLevel("ERROR")
PATH = os.path.abspath("web_synth.txt")

def spark_pr(n, d=0.85, tol=1e-6, max_iter=100):
    t0 = time.time()
    lines = sc.textFile(PATH, 4).filter(lambda l: not l.startswith("#"))
    edges = (lines.map(lambda l: tuple(map(int, l.split("\t"))))
                  .filter(lambda e: e[0] != e[1] and e[0] < n and e[1] < n).distinct())
    links = edges.groupByKey().mapValues(list).cache()
    allnodes = sc.parallelize(range(n), 4).map(lambda i: (i, 0.0))
    dang = allnodes.subtractByKey(links).cache()
    nlinks = links.count(); ndang = dang.count()
    ranks = allnodes.mapValues(lambda _: 1.0/n).cache(); ranks.count()
    prep = time.time() - t0
    t1 = time.time(); errs = []
    for it in range(1, max_iter+1):
        contribs = links.join(ranks).flatMap(lambda x: [(j, x[1][1]/len(x[1][0])) for j in x[1][0]])
        summed = contribs.reduceByKey(add)
        dm = dang.join(ranks).map(lambda x: x[1][1]).sum()
        new = (allnodes.leftOuterJoin(summed)
                       .mapValues(lambda v: d*((v[1] or 0.0) + dm/n) + (1-d)/n)).cache()
        err = new.join(ranks).map(lambda x: abs(x[1][0]-x[1][1])).sum()
        errs.append(err); ranks.unpersist(); ranks = new
        if err < tol: break
    out = np.zeros(n)
    for k, v in ranks.collect(): out[k] = v
    iters_t = time.time() - t1
    ranks.unpersist(); links.unpersist(); dang.unpersist()
    return out, dict(prep=prep, iterate=iters_t, iters=it, errs=errs, dangling=ndang)

spark_pr(2000)  # warm-up (JVM, workers)
rows = []; full = None
for n in [10_000, 25_000, 50_000, 100_000]:
    E, N, pre = load_clean(nmax=n)
    # NumPy/SciPy
    t=time.time(); M, outdeg = build_matrix(E, N); tb=time.time()-t
    t=time.time(); r_np, it_np, errs = pagerank_numpy(E, N); tt=time.time()-t   # includes its own build
    t_np_iter = tt - tb
    # NetworkX
    t=time.time(); G = nx.DiGraph(); G.add_nodes_from(range(N)); G.add_edges_from(map(tuple, E.tolist())); nb=time.time()-t
    t=time.time(); pr = nx.pagerank(G, alpha=0.85, tol=1e-6/N, max_iter=200); ni=time.time()-t
    r_nx = np.array([pr[i] for i in range(N)])
    # Spark
    r_sp, info = spark_pr(N)
    row = dict(n=n, edges=int(len(E)),
        numpy=dict(build=tb, iterate=max(t_np_iter,0), iters=it_np),
        networkx=dict(build=nb, iterate=ni),
        spark=dict(prep=info["prep"], iterate=info["iterate"], iters=info["iters"], dangling=info["dangling"]),
        max_diff_spark_numpy=float(np.abs(r_sp-r_np).max()), l1_spark_numpy=float(np.abs(r_sp-r_np).sum()),
        top20_spark_numpy=len(set(np.argsort(-r_sp)[:20])&set(np.argsort(-r_np)[:20])),
        spearman_spark_numpy=float(spearmanr(r_sp, r_np)[0]), spark_sum=float(r_sp.sum()))
    if n == 100_000: row["spark_errs"] = info["errs"]; full = r_sp
    rows.append(row); print(json.dumps({k:v for k,v in row.items() if k!="spark_errs"}), flush=True)
json.dump(rows, open("results_scale.json","w"), indent=1)
spark.stop()
