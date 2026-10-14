"""ci_lowrank.py — python ci_lowrank.py N_BOOT RESULTS_DIR
Core-cluster bootstrap CI for CV R² of subst-only / additive / rank1-3 (cell-grouped folds so duplicated cells never leak)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RD = sys.argv[2] if len(sys.argv) > 2 else '.'
from d1s_lowrank import als, additive, subst_only
P=pd.read_csv(os.path.join(RD,'pairs_H.csv')); P=P[~P.q_bad].copy()
P['ctx']=P.core+'|'+P.orient.fillna('')+'|'+P.face.fillna('')+'|'+P.stereo_index.astype(str)
def cv_grouped(M,mask,cellid,fitter,nfold=5,seed=1):
    rng=np.random.default_rng(seed); ids=np.unique(cellid[mask]); rng.shuffle(ids)
    fold_of={c:i%nfold for i,c in enumerate(ids)}
    F=np.full(M.shape,-1); F[mask]=[fold_of[c] for c in cellid[mask]]
    yall=M[mask]; sse=sst=0.0
    for f in range(nfold):
        te=(F==f); tr=mask&~te
        pred=fitter(M,tr); y=M[te]; p=pred[te]
        sse+=((y-p)**2).sum(); sst+=((y-yall.mean())**2).sum()
    return 1-sse/sst
FIT={'subst':subst_only,'additive':additive,'rank1':lambda M,m:als(M,m,1)[0],'rank2':lambda M,m:als(M,m,2)[0],'rank3':lambda M,m:als(M,m,3)[0]}
def build(sub,col,cores):
    rows=[];cells=[]
    for bi,k in enumerate(cores):
        g=sub[sub.core==k]
        piv=g.pivot_table(index='ctx',columns='R',values=col,aggfunc='mean')
        for ctx,r in piv.iterrows(): rows.append(r.rename(f'{bi}|{ctx}')); cells.append(ctx)
    D=pd.DataFrame(rows); Xs=sorted(sub.R.unique()); D=D.reindex(columns=Xs)
    M=D.values; mask=~np.isnan(M)
    cellid=np.array([[f'{c}|{x}' for x in Xs] for c in cells],dtype=object)
    return np.nan_to_num(M),mask,cellid
jobs=[('dipole','barrier_kcal'),('dipolarophile','barrier_kcal'),('dipole','pauli_dft')]
NB=int(sys.argv[1]) if len(sys.argv)>1 else 200
out=[]
for side,col in jobs:
    sub=P[(P.side==side)&(P.attach=='direct')]
    cores=np.array(sorted(sub.core.unique())); rng=np.random.default_rng(11)
    for b in range(NB+1):
        cs=cores if b==0 else rng.choice(cores,len(cores))
        M,mask,cid=build(sub,col,cs)
        r={'side':side,'ch':col,'boot':b}
        for k,f in FIT.items(): r[k]=cv_grouped(M,mask,cid,f)
        out.append(r)
        if b%20==0: print(side,col,b,{k:round(v,3) for k,v in r.items() if k in FIT},flush=True); pd.DataFrame(out).to_csv(os.path.join(RD,'ci_lowrank_boot.csv'),index=False)
pd.DataFrame(out).to_csv(os.path.join(RD,'ci_lowrank_boot.csv'),index=False)
print('DONE')
