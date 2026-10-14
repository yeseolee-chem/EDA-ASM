import sys, os, numpy as np, pandas as pd
LAB, RD = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1s_prep import load_labels, budget
df=load_labels(LAB)
G=pd.read_csv(os.path.join(RD,'presence_proxy.csv'))
MAN={'nitrile oxide':2096,'azide':2096,'azomethine ylide':2334,'nitrone':2736,'carbonyl ylide':2334}  # SPEC_D1_RUN §3.3
pres=df.groupby('dipole').size()
clean=df[~df.q_bad].groupby('dipole').size()
P=pd.read_csv(os.path.join(RD,'pairs_H.csv')); Pc=P[~P.q_bad]
neg=df[~df.q_bad].groupby('dipole').apply(lambda g:(g.barrier_kcal<0).mean())
par_cores=df[df.is_parent].groupby('dipole').core.nunique(); all_cores=df.groupby('dipole').core.nunique()
T=pd.DataFrame({'manifest':pd.Series(MAN),'present':pres,'present_frac':pres/pd.Series(MAN),'clean':clean,'pairs_clean':Pc.groupby('dipole').size(),'cores':all_cores,'cores_with_H_parent':par_cores,'neg_barrier_frac':neg})
print(T.round(3).to_string())
cy=G[G.dipole=='carbonyl ylide']
print('\ncarbonyl ylide presence by skeleton x series (proxy):')
print(cy.groupby(['skeleton','series']).apply(lambda g: f"{int(g.n.sum())}/{int(g.n_sib.sum())}").unstack().to_string())
print('\ncarbonyl ylide presence by X (direct, proxy):')
x=cy[~cy.R.str.match(r'^[pmo]-')].groupby('R').apply(lambda g:g.n.sum()/g.n_sib.sum()).sort_values()
print(x.round(2).to_string())
print('\nbarrier of carbonyl ylide present rows (clean):', df[(~df.q_bad)&(df.dipole=='carbonyl ylide')].barrier_kcal.describe().round(1).to_dict())
# sensitivity: headline with vs without carbonyl ylide
def head(Pc,R,E):
    INT=['elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft','bsse_shift_kcal']
    y=Pc.barrier_kcal; st=Pc.d1_kcal+Pc.d2_kcal; it=Pc[INT].sum(axis=1); b=budget(Pc)
    w=np.sign(R.barrier_kcal)
    return dict(n_pairs=len(Pc),r_strain=np.corrcoef(st,y)[0,1],r_Eint=np.corrcoef(it,y)[0,1],sh_Pauli=b['Pauli'],sh_elst=b['elst'],sh_OI=b['OI'],sh_strain=b['d1']+b['d2'],
                regio_acc_strain=(np.sign(R.d1_kcal+R.d2_kcal)==w).mean(),regio_acc_Eint=(np.sign(R[INT].sum(axis=1))==w).mean(),exo_frac=(E.barrier_kcal<0).mean(),n_regio=len(R),n_exo=len(E))
R=pd.read_csv(os.path.join(RD,'sel_regio.csv')); E=pd.read_csv(os.path.join(RD,'sel_exoendo.csv'))
S=pd.DataFrame({'with carbonyl ylide':head(Pc,R,E),'without':head(Pc[Pc.dipole!='carbonyl ylide'],R[R.dipole!='carbonyl ylide'],E[E.dipole!='carbonyl ylide'])})
print('\n',S.round(3).to_string())
S.to_csv(os.path.join(RD,'q4_sensitivity.csv')); T.to_csv(os.path.join(RD,'q4_missing_by_dipole.csv'))
