"""hammett_eval.py — Hammett-level evaluation of pair predictions (same bend rule as the D1 analysis, hidden_kink.py).

series of one para family: points (sigma_p of X, y_X) with y = -dd / 1.3642 (log k units) plus H at (0, 0)
bend call: one hinge at psi between neighbouring sigma values, >= 3 points each side,
           AICc(line) - AICc(hinge) >= 4, and still >= 2 after removing any one non-H point
V call:    bend call and the two arms have opposite signs, both |arm| >= 0.46
usage (self-test on DFT values): python -I hammett_eval.py pairs_d1.csv"""
import sys, numpy as np, pandas as pd

K = 1.3642
CH = ["barrier", "d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds"]


def aicc(rss, n, k):
    return n * np.log(np.maximum(rss, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def bent_gain(x, Y):
    """x (n,), Y (n, m) -> gain (m,), psi (m,), left slope (m,), right slope (m,)"""
    n = len(x)
    A = np.c_[np.ones(n), x]
    rl = ((Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]) ** 2).sum(0)
    xs = np.unique(x); best = np.full(Y.shape[1], np.inf); psi = np.full(Y.shape[1], np.nan)
    L = np.full(Y.shape[1], np.nan); R = np.full(Y.shape[1], np.nan)
    for i in range(len(xs) - 1):
        p = (xs[i] + xs[i + 1]) / 2
        if (x < p).sum() < 3 or (x > p).sum() < 3:
            continue
        B = np.c_[np.ones(n), x, np.maximum(0, x - p)]
        C = np.linalg.lstsq(B, Y, rcond=None)[0]
        rss = ((Y - B @ C) ** 2).sum(0); better = rss < best
        best[better] = rss[better]; psi[better] = p; L[better] = C[1, better]; R[better] = C[1, better] + C[2, better]
    return aicc(rl, n, 3) - aicc(best, n, 5), psi, L, R


def calls(x, Y, names):
    """bend and V calls for every column of Y; names = point labels ('H' is never removed)"""
    g, psi, L, R = bent_gain(x, Y)
    ok = g >= 4
    for i, nm in enumerate(names):
        if nm == "H" or not ok.any():
            continue
        keep = np.arange(len(x)) != i
        g2, *_ = bent_gain(x[keep], Y[keep][:, ok])
        ok[np.where(ok)[0][g2 < 2]] = False
    v = ok & (L * R < 0) & (np.abs(L) >= 0.46) & (np.abs(R) >= 0.46)
    return ok, v, g, psi, L, R


def family_series(P, fam, col, min_x=12):
    """x (sigma_p) and y (log units) of one para family from pair rows P (column col = dd in kcal); H appended"""
    t = P[(P.fam == fam) & (P.pos == "p") & P.sigma.notna()].drop_duplicates("Xb")
    if len(t) + 1 < min_x:
        return None
    x = np.r_[t.sigma.to_numpy(float), 0.0]; y = np.r_[-t[col].to_numpy(float) / K, 0.0]
    return x, y, list(t.Xb) + ["H"]


def rho_r2(x, y):
    c = np.polyfit(x, y, 1); r2 = 1 - ((y - np.polyval(c, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return c[0], r2


if __name__ == "__main__":
    P = pd.read_csv(sys.argv[1])
    fams = sorted(P[P.pos == "p"].fam.unique()); rows = []
    for f in fams:
        s = family_series(P, f, "dd_barrier")
        if s is None:
            continue
        x, _, names = s
        Y = np.c_[[family_series(P, f, f"dd_{c}")[1] for c in CH]].T
        ok, v, *_ = calls(x, Y, names)
        rows.append(dict(fam=f, **{f"bent_{c}": bool(ok[k]) for k, c in enumerate(CH)}, **{f"V_{c}": bool(v[k]) for k, c in enumerate(CH)}))
    T = pd.DataFrame(rows)
    print("para families evaluated:", len(T))
    print("DFT bend calls per target:", {c: int(T[f"bent_{c}"].sum()) for c in CH})
    print("DFT V calls per target:", {c: int(T[f"V_{c}"].sum()) for c in CH})
    hid = (~T.bent_barrier) & T[[f"bent_{c}" for c in CH[1:]]].any(axis=1)
    print("barrier straight & >= 1 channel bent:", int(hid.sum()))
