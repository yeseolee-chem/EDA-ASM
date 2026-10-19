import os, sys, tempfile, numpy as np
sys.path.insert(0, os.environ["CODE"])
import d1_train as T
from joblib import Parallel, delayed
rng = np.random.default_rng(0)
zd, zm = rng.normal(size=(300, 10)), rng.normal(size=(300, 12))
td, tm = rng.normal(size=(80, 10)), rng.normal(size=(80, 12))
ys, yte = rng.normal(size=300), rng.normal(size=80)
with tempfile.TemporaryDirectory() as d:
    T._MMDIR.append(d)
    Dtr, Dte = T.mm(T.dist3(zd, zm, zd, zm)), T.mm(T.dist3(td, tm, zd, zm))
    print(type(Dtr[0]).__name__, Dtr[0].shape)
    a = Parallel(n_jobs=2, max_nbytes="1M")(delayed(T._krr_cell)(Dtr, Dte, ys, yte, 1.0, 1e-2, 1e-2, [1e-3, 1e-1]) for _ in range(2))
    b = Parallel(n_jobs=2, max_nbytes="1M")(delayed(T._svr_one)(Dtr, Dte, ys, yte, 1.0, 1e-2, 1e-2, 10, 0.1) for _ in range(2))
    c = T._svr_cell(T.dist3(zd, zm, zd, zm), T.dist3(td, tm, zd, zm), ys, yte, 1.0, 1e-2, 1e-2, [10], [0.1])
    k = T._krr_cell(T.dist3(zd, zm, zd, zm), T.dist3(td, tm, zd, zm), ys, yte, 1.0, 1e-2, 1e-2, [1e-3, 1e-1])
    print("krr", a, k); print("svr", b, c)
    assert a[0] == k and b[0] == c[0], "memmap path differs"
print("smoke ok")
