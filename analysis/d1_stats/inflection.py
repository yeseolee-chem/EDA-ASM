"""inflection.py — EDG→EWG 순서(σ)에서 채널별 변곡점(기울기 변화) 탐지 + 도너/억셉터 비대칭(Houk vs Domingo).

python inflection.py LABELS RESULTS_DIR SIGMA_CSV OUTDIR
  SIGMA_CSV 열: X, sigma_m, sigma_p (Hansch, Leo & Taft 1991 원표 값; 빈칸 허용)
계열 = (dipole, 골격, 쪽, 부착, 문맥[orient|face|stereo]). direct·para는 σp, meta는 σm, ortho는 제외.
  direct: H 부모 대비 ΔΔ (pairs_H.csv).  para/meta: Ph 대비 ΔΔ (pairs_aryl.csv) — Hammett 본래 설정.
모형 (각 계열 × 타깃): L 선형 / S 연속 꺾은선(꺾임점 1개, 격자 탐색) / Q 이차.
변곡 판정(사전 고정): AICc(S) ≤ AICc(L) − 4 이고 LOO RMSE(S) < LOO RMSE(L). 계열당 치환기 ≥ 12.
거짓 양성률: 계열 안에서 σ를 섞은 순열 200회로 같은 판정을 돌린 비율(경험적 귀무).
비대칭: 꺾임점을 σ = 0에 고정한 꺾은선의 기울기 b_donor(σ<0 쪽), b_acceptor(σ>0 쪽).
"""
import sys, os, numpy as np, pandas as pd
LAB, RD, SIG, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
MIN_N = 12; DAICC = 4.0; NPERM = 200; SEED = 20261014
TGT = ['barrier_kcal', 'd1_kcal', 'd2_kcal', 'elst_dft', 'pauli_dft', 'oi_dft', 'disp_dft', 'cpcm_dft', 'cds_dft']
INT = ['elst_dft', 'pauli_dft', 'oi_dft', 'disp_dft', 'cpcm_dft', 'cds_dft', 'bsse_shift_kcal']
sg = pd.read_csv(SIG).set_index('X')

def design(x, kind, bp=None):
    one = np.ones_like(x)
    if kind == 'L': return np.c_[one, x]
    if kind == 'Q': return np.c_[one, x, x ** 2]
    return np.c_[one, x, np.maximum(x - bp, 0.0)]

def fit(x, y, kind, bp=None):
    X = design(x, kind, bp); b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    return b, (r ** 2).sum()

def aicc(rss, n, k):
    return n * np.log(rss / n + 1e-12) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)

def best_seg(x, y):
    grid = np.unique(np.quantile(x, np.linspace(0.2, 0.8, 13)))
    best = None
    for bp in grid:
        b, rss = fit(x, y, 'S', bp)
        if best is None or rss < best[1]: best = (bp, rss, b)
    return best

def loo(x, y, kind, bp=None):
    e = []
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        if kind == 'S':
            bpi = best_seg(x[m], y[m])[0]; b, _ = fit(x[m], y[m], 'S', bpi); e.append(y[i] - (design(x[i:i+1], 'S', bpi) @ b)[0])
        else:
            b, _ = fit(x[m], y[m], kind); e.append(y[i] - (design(x[i:i+1], kind) @ b)[0])
    return np.sqrt(np.mean(np.square(e)))

def judge(x, y):
    n = len(x); _, rL = fit(x, y, 'L'); bp, rS, bS = best_seg(x, y); _, rQ = fit(x, y, 'Q')
    aL, aS, aQ = aicc(rL, n, 2), aicc(rS, n, 4), aicc(rQ, n, 3)
    infl = (aS <= aL - DAICC)
    if infl: infl = loo(x, y, 'S') < loo(x, y, 'L')
    return dict(aicc_L=aL, aicc_S=aS, aicc_Q=aQ, bp=bp, slope_left=bS[1], slope_right=bS[1] + bS[2], inflection=bool(infl))

P = pd.read_csv(os.path.join(RD, 'pairs_H.csv')); P = P[~P.q_bad & (P.attach == 'direct')].copy(); P['pos'] = 'direct'
A = pd.read_csv(os.path.join(RD, 'pairs_aryl.csv')); A = A[A.pos.isin(['p', 'm'])].copy()
for D in (P, A):
    D['strain'] = D.d1_kcal + D.d2_kcal; D['Eint'] = D[INT].sum(axis=1); D['nosolv'] = D.barrier_kcal - D.cpcm_dft - D.cds_dft
P['series'] = P.dipole + '|' + P.skeleton + '|' + P.side + '|direct|' + P.ctx.astype(str)
A['series'] = A.dipole + '|' + A.skeleton + '|' + A.side + '|' + A.pos + '|' + A.ctx.astype(str)
D = pd.concat([P, A], ignore_index=True)
D['sigma'] = np.where(D.pos == 'm', D.X.map(sg['sigma_m']), D.X.map(sg['sigma_p']))
D = D.dropna(subset=['sigma'])
targets = TGT + ['strain', 'Eint', 'nosolv']
rng = np.random.default_rng(SEED); rows = []; null = []
for s, g in D.groupby('series'):
    g = g.groupby('X').mean(numeric_only=True)        # one value per substituent (siblings averaged)
    if len(g) < MIN_N: continue
    x = g.sigma.values
    dp, sk, side, pos = s.split('|')[:4]
    for t in targets:
        y = g[t].values
        r = judge(x, y); r.update(series=s, dipole=dp, skeleton=sk, side=side, pos=pos, target=t, n=len(g))
        # fixed breakpoint at sigma = 0: donor vs acceptor slopes
        if (x < 0).sum() >= 3 and (x > 0).sum() >= 3:
            b0, _ = fit(x, y, 'S', 0.0); r['slope_donor'] = b0[1]; r['slope_acceptor'] = b0[1] + b0[2]
        rows.append(r)
        if t in ('barrier_kcal', 'elst_dft', 'pauli_dft', 'oi_dft'):
            for _ in range(NPERM // 10):            # 20 permutations per series/target -> pooled null
                null.append(judge(rng.permutation(x), y)['inflection'])
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'inflection_series.csv'), index=False)
summ = R.groupby('target').agg(n_series=('inflection', 'size'), frac_inflection=('inflection', 'mean'), median_bp=('bp', 'median'))
summ['null_rate_permutation'] = np.mean(null) if null else np.nan
# hidden inflections: channel bends while the barrier does not
piv = R.pivot_table(index='series', columns='target', values='inflection')
hid = {t: float(((piv[t] == 1) & (piv['barrier_kcal'] == 0)).mean()) for t in targets if t != 'barrier_kcal'}
summ['hidden_frac_vs_barrier'] = pd.Series(hid)
summ.to_csv(os.path.join(OUT, 'inflection_summary.csv'))
asym = R.dropna(subset=['slope_donor']).groupby(['dipole', 'skeleton', 'side', 'pos', 'target'])[['slope_donor', 'slope_acceptor']].median()
asym.to_csv(os.path.join(OUT, 'asymmetry_slopes.csv'))
print(summ.round(3).to_string())
