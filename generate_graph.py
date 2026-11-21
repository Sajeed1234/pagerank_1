import numpy as np
rng = np.random.default_rng(42)
N = 100_000
M = 8
buf = np.zeros(N*12 + 100, dtype=np.int64)
blen = M
buf[:M] = np.arange(M)
src = []; dst = []
for i in range(M, N):
    if rng.random() < 0.10:
        buf[blen] = i; blen += 1; continue
    k = int(rng.integers(3, 2*M-2))
    pref = rng.random(k) < 0.8
    t = np.where(pref, buf[rng.integers(0, blen, k)], rng.integers(0, i, k))
    src.extend([i]*k); dst.extend(t.tolist())
    buf[blen:blen+k] = t; blen += k
    buf[blen] = i; blen += 1
E = np.stack([np.array(src), np.array(dst)], 1)
selfl = np.stack([np.arange(0, 2000), np.arange(0, 2000)], 1)
dup = E[rng.integers(0, len(E), 5000)]
E = np.vstack([E, selfl, dup])
rng.shuffle(E)
np.savetxt("web_synth.txt", E, fmt="%d", delimiter="\t",
           header="Synthetic directed web graph\nFromNodeId\tToNodeId", comments="# ")
print("raw edges", len(E), "nodes", N)
