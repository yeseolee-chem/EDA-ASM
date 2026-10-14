"""q5q7.py — dipole별 E_int–장벽 상관(CI, 강건성, Δζ 통제)과 TS 위치를 맞춘 regio 형제 분석.
python q5q7.py LABELS RESULTS_DIR   (RESULTS_DIR = run_all.py 출력 폴더)"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LAB, RD = sys.argv[1], sys.argv[2]
from d1s_prep import load_labels
df=load_labels(LAB)
P=pd.read_csv(os.path.join(RD,'pairs_H.csv'))
INT=['elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft','bsse_shift_kcal']
alt=df.set_index('job_id').alt_used
P['alt_same']=P.job_id.map(alt).values==P.parent_job.map(alt).values
P['Eint']=P[INT].sum(axis=1); P['strain']=P.d1_kcal+P.d2_kcal
rng=np.random.default_rng(3)
def r_ci(g,B=1000):
    r0=np.corrcoef(g.Eint,g.barrier_kcal)[0,1]; cores=g.core.unique(); byc={k:x for k,x in g.groupby('core')}; bs=[]
    for _ in range(B):
        D=pd.concat([byc[k] for k in rng.choice(cores,len(cores))]); bs.append(np.corrcoef(D.Eint,D.barrier_kcal)[0,1])
    lo,hi=np.percentile(bs,[2.5,97.5]); return r0,lo,hi,len(g),len(cores)
print('=== Q5: r(ΔΔEint, ΔΔE‡) by dipole, core-bootstrap 95% CI')
variants={'base (clean)':P[~P.q_bad],'incl. stepwise':P,'alt_used same':P[(~P.q_bad)&P.alt_same]}
# min-barrier sibling only (per cell, regio, face) on clean
c=P[~P.q_bad].copy(); c['key']=c.cell_id+'|'+c.regio+'|'+c.face.fillna('')
variants['lowest sibling only']=c.loc[c.groupby('key').barrier_kcal.idxmin()]
rows=[]
for vn,V in variants.items():
    for dp,g in V.groupby('dipole'):
        r0,lo,hi,n,nc=r_ci(g,500 if vn!='base (clean)' else 1000); rows.append(dict(variant=vn,dipole=dp,n=n,n_core=nc,r=r0,lo=lo,hi=hi))
Q5=pd.DataFrame(rows); print(Q5.round(2).pivot_table(index='dipole',columns='variant',values=['r']).round(2).to_string())
print(Q5[Q5.variant=='base (clean)'].round(2).to_string(index=False))
Q5.to_csv(os.path.join(RD,'q5_dipole_eint.csv'),index=False)
# by side within dipole
Pc=P[~P.q_bad]
print(Pc.groupby(['dipole','side']).apply(lambda g: pd.Series({'n':len(g),'r_Eint':np.corrcoef(g.Eint,g.barrier_kcal)[0,1],'r_strain':np.corrcoef(g.strain,g.barrier_kcal)[0,1],'r_Eint_dzeta':np.corrcoef(g.Eint,g.zeta)[0,1]})).round(2).to_string())
# partial: Eint vs barrier controlling Δζ, Δasyn, per dipole
from sklearn.linear_model import LinearRegression
def resid(y,X): return y-LinearRegression().fit(X,y).predict(X)
print('partial r(Eint, ΔE‡ | Δζ, Δasyn) by dipole')
for dp,g in Pc.groupby('dipole'):
    X=g[['zeta','asyn']].values; print(dp, round(np.corrcoef(resid(g.Eint.values,X),resid(g.barrier_kcal.values,X))[0,1],2))
print()
print('=== Q7: regio pairs, controlling TS position')
R=pd.read_csv(os.path.join(RD,'sel_regio.csv')); R['Eint']=R[INT].sum(axis=1); R['strain']=R.d1_kcal+R.d2_kcal
w=np.sign(R.barrier_kcal)
for thr in [None,0.05,0.03,0.02]:
    S=R if thr is None else R[R.zeta.abs()<thr]
    ws=np.sign(S.barrier_kcal)
    print(f'|Δζ|<{thr}', 'n',len(S), 'acc strain',round((np.sign(S.strain)==ws).mean(),2),'acc Eint',round((np.sign(S.Eint)==ws).mean(),2),'acc OI',round((np.sign(S.oi_dft)==ws).mean(),2),'acc elst',round((np.sign(S.elst_dft)==ws).mean(),2),'acc Pauli',round((np.sign(S.pauli_dft)==ws).mean(),2))
X=R[['zeta','asyn']].values
for nm in ['Eint','strain','oi_dft','elst_dft','pauli_dft']:
    print('partial r(Δ',nm,', ΔE‡ | Δζ,Δasyn) =', round(np.corrcoef(resid(R[nm].values,X),resid(R.barrier_kcal.values,X))[0,1],3), ' raw r =', round(np.corrcoef(R[nm],R.barrier_kcal)[0,1],3))
# bootstrap CI of acc in |Δζ|<0.03 subset
S=R[R.zeta.abs()<0.03]; cores=S.core.unique(); byc={k:x for k,x in S.groupby('core')}; bs=[]
for _ in range(1000):
    D=pd.concat([byc[k] for k in rng.choice(cores,len(cores))]); ws=np.sign(D.barrier_kcal)
    bs.append(((np.sign(D.strain)==ws).mean(),(np.sign(D.Eint)==ws).mean()))
bs=np.array(bs); print('|Δζ|<0.03 CI strain',np.percentile(bs[:,0],[2.5,97.5]).round(2),'Eint',np.percentile(bs[:,1],[2.5,97.5]).round(2), 'n cores',len(cores))
print(S.groupby('dipole').apply(lambda g: pd.Series({'n':len(g),'acc_strain':(np.sign(g.strain)==np.sign(g.barrier_kcal)).mean(),'acc_Eint':(np.sign(g.Eint)==np.sign(g.barrier_kcal)).mean()})).round(2).to_string())
