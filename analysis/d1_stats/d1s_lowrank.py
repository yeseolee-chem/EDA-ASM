"""d1s_lowrank.py — 결측이 있는 (context × substituent) 행렬의 저랭크(σ·ρ) 적합과 교차검증.

ΔΔ는 H에서 0이므로 절편 없는 M ≈ U Vᵀ (rank k)가 Hammett형 σ_X·ρ_ctx 모형이다(k = 1).
"""
import numpy as np

LAM = 1.0      # ridge on factors (fixed; pre-registered)
ITERS = 60
NFOLD = 5
SEED = 1


def als(M, mask, k, lam=LAM, iters=ITERS, seed=0):
    n, m = M.shape
    if k == 0:
        return np.zeros_like(M), None, None
    rng = np.random.default_rng(seed)
    U = rng.normal(scale=0.1, size=(n, k))
    V = rng.normal(scale=0.1, size=(m, k))
    Mz = np.where(mask, M, 0.0)
    I = lam * np.eye(k)
    for _ in range(iters):
        for i in range(n):
            o = mask[i]
            U[i] = np.linalg.solve(V[o].T @ V[o] + I, V[o].T @ Mz[i, o])
        for j in range(m):
            o = mask[:, j]
            V[j] = np.linalg.solve(U[o].T @ U[o] + I, U[o].T @ Mz[o, j])
    return U @ V.T, U, V


def additive(M, mask):
    n, m = M.shape
    a, b = np.zeros(n), np.zeros(m)
    mu = M[mask].mean()
    for _ in range(100):
        a = np.nan_to_num(np.nanmean(np.where(mask, M - mu - b[None, :], np.nan), 1))
        b = np.nan_to_num(np.nanmean(np.where(mask, M - mu - a[:, None], np.nan), 0))
    return mu + a[:, None] + b[None, :]


def subst_only(M, mask):
    b = np.nan_to_num(np.nanmean(np.where(mask, M, np.nan), 0))
    return np.tile(b, (M.shape[0], 1))


def cv(M, mask, fitter, nfold=NFOLD, seed=SEED):
    """mask된 칸을 무작위 5등분해 하나씩 가리고 예측. 반환: (CV R², CV MAE)."""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    rng.shuffle(idx)
    yall = M[mask]
    sse = sst = ae = 0.0
    cnt = 0
    for f in np.array_split(idx, nfold):
        tr = mask.copy()
        tr[f[:, 0], f[:, 1]] = False
        pred = fitter(M, tr)
        y = M[f[:, 0], f[:, 1]]
        p = pred[f[:, 0], f[:, 1]]
        sse += ((y - p) ** 2).sum()
        sst += ((y - yall.mean()) ** 2).sum()
        ae += np.abs(y - p).sum()
        cnt += len(y)
    return 1 - sse / sst, ae / cnt


def matrix(P, value, index="ctx", columns="R"):
    piv = P.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    M = piv.values
    mask = ~np.isnan(M)
    return np.nan_to_num(M), mask, piv
