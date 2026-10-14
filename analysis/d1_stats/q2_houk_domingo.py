import sys, os, numpy as np, pandas as pd
LAB, RD = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1s_prep import load_labels
df=load_labels(LAB)
d=df[~df.q_bad].copy()
P=pd.read_csv(os.path.join(RD,'pairs_H.csv')); P=P[~P.q_bad]
INT=['elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft','bsse_shift_kcal']
P['solv']=P.cpcm_dft+P.cds_dft; P['strain']=P.d1_kcal+P.d2_kcal; P['Eint']=P[INT].sum(axis=1)
P['nosolv']=P.barrier_kcal-P.solv
# Domingo's categories (his own classification of these alkenes): vinyl amines = supernucleophiles; nitroethylene, acrylonitrile = strong electrophiles
DON=['NMe2','NH2']; ACC=['NO2','CN']; MID=['OMe','OH','CO2Me','CHO','Ac','CF3','Me','Cl','F','Br']
rows=[]
for dp in ['azide','nitrile oxide','nitrone']:
    for sk in ['A','B','C','D']:
        s=P[(P.dipole==dp)&(P.side=='dipolarophile')&(P.attach=='direct')&(P.skeleton==sk)]
        if s.empty: continue
        # kinetically relevant TS per X: lowest barrier over regio/stereo
        best=s.loc[s.groupby('X').barrier_kcal.idxmin()].set_index('X')
        for X in DON+ACC+['OMe','CO2Me','Me']:
            if X in best.index:
                b=best.loc[X]; rows.append(dict(dipole=dp,skel=sk,X=X,ddE=b.barrier_kcal,dd_strain=b.strain,dd_Eint=b.Eint,dd_solv=b.solv,dd_nosolv=b.nosolv,dd_elst=b.elst_dft,dd_pauli=b.pauli_dft,dd_oi=b.oi_dft))
T=pd.DataFrame(rows)
pd.set_option('display.width',250)
print(T.pivot_table(index=['dipole','X'],columns='skel',values='ddE').round(1).to_string())
print('\n-- solvation-stripped (ΔΔE‡ − ΔΔ(CPCM+CDS))')
print(T.pivot_table(index=['dipole','X'],columns='skel',values='dd_nosolv').round(1).to_string())
# asymmetry index per dipole/skeleton: mean(donor lowering) / mean(acceptor lowering)
print('\n-- asymmetry: mean ΔΔE‡ donors (NMe2,NH2) vs acceptors (NO2,CN)')
for (dp,sk),g in T.groupby(['dipole','skel']):
    dn=g[g.X.isin(DON)].ddE.mean(); ac=g[g.X.isin(ACC)].ddE.mean(); dn2=g[g.X.isin(DON)].dd_nosolv.mean(); ac2=g[g.X.isin(ACC)].dd_nosolv.mean()
    print(f'{dp:14s} {sk}  donors {dn:+.1f}  acceptors {ac:+.1f}   | no-solv: donors {dn2:+.1f} acceptors {ac2:+.1f}')
# regioselectivity magnitude for azide R2 by X
R=pd.read_csv(os.path.join(RD,'sel_regio.csv'))
R=R[(R.series=='R2')&(R.attach=='direct')]
print('\n-- |ΔE‡(regio a−b)| (kcal/mol), median over contexts, by dipole and X')
print(R.assign(absd=R.barrier_kcal.abs()).pivot_table(index='X',columns='dipole',values='absd',aggfunc='median').round(1).loc[DON+['OMe','OH']+ACC+['CO2Me','CHO','Ac','CF3']].to_string())
T.to_csv(os.path.join(RD,'q2_azide_table.csv'),index=False)
