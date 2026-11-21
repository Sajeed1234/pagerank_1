import sys, resource, numpy as np, json
from pagerank_lib import *
m = sys.argv[1]
if m == "save":
    E, N, _ = load_clean(); np.save("E_clean.npy", E); sys.exit()
E = np.load("E_clean.npy"); N = 100000
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
if m == "numpy": pagerank_numpy(E, N)
else: pagerank_nx(E, N)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
print(json.dumps(dict(method=m, base_mb=round(base,1), peak_mb=round(peak,1), extra_mb=round(peak-base,1))))
