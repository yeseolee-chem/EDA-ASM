import sys, os, numpy as np, pandas as pd
LAB, RD = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1s_prep import load_labels, CH
df=load_labels(LAB)
d=df[(~df.q_bad)&(df.series=='R1')].copy()      # dipole substituent varies, dipolarophile skeleton fixed (R2 = H or '-')
d['strain']=d.d1_kcal+d.d2_kcal
INT=['elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft','bsse_shift_kcal']
d['Eint']=d[INT].sum(axis=1)
COLS=['barrier_kcal','strain','Eint','d1_kcal','d2_kcal','elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft','cds_dft']
# kinetically relevant TS per (dipole, R1, skeleton[, orient]): lowest barrier among stereo siblings (and faces)
best=d.loc[d.groupby(['dipole','R1','skeleton']).barrier_kcal.idxmin()].set_index(['dipole','R1','skeleton'])
bestO=d[d.orient.isin(['a','b'])]
bestO=bestO.loc[bestO.groupby(['dipole','R1','skeleton','orient']).barrier_kcal.idxmin()].set_index(['dipole','R1','skeleton','orient'])
COMP=[('Ap','A','gem-Me2 alkene (2-methylpropene − ethylene)'),('AppE','A','trans-2-butene − ethylene'),('AppZ','A','cis-2-butene − ethylene'),
      ('AppZ','AppE','cis − trans 2-butene'),('Cp','C','2-butyne − acetylene'),('Bp','B','2,3-diMe-norbornene − norbornene'),
      ('Dp','D','3,3-diMe-cyclooctyne − cyclooctyne'),('DppC','D','cis-3,8-diMe − cyclooctyne'),('DppT','D','trans-3,8-diMe − cyclooctyne'),('DppC','DppT','cis − trans 3,8-diMe-cyclooctyne')]
rng=np.random.default_rng(5); rows=[]
for s1,s0,lab in COMP:
    for dp in sorted(d.dipole.unique()):
        try:
            A=best.xs(s1,level='skeleton').loc[dp]; B=best.xs(s0,level='skeleton').loc[dp]
        except KeyError: continue
        common=A.index.intersection(B.index)
        if len(common)<8: rows.append(dict(comp=lab,dipole=dp,n=len(common))); continue
        D=A.loc[common,COLS]-B.loc[common,COLS]
        r=dict(comp=lab,dipole=dp,n=len(common))
        for c in COLS:
            v=D[c].values; bs=[np.median(rng.choice(v,len(v))) for _ in range(500)]
            r[c]=np.median(v); r[c+'_lo'],r[c+'_hi']=np.percentile(bs,[2.5,97.5])
        r['frac_slower']=(D.barrier_kcal>0).mean()
        rows.append(r)
M=pd.DataFrame(rows); M.to_csv(os.path.join(RD,'q5_methyl.csv'),index=False)
show=['comp','dipole','n','barrier_kcal','barrier_kcal_lo','barrier_kcal_hi','frac_slower','strain','Eint','d1_kcal','d2_kcal','elst_dft','pauli_dft','oi_dft','disp_dft','cpcm_dft']
pd.set_option('display.width',300); pd.set_option('display.max_columns',30)
print(M[show].round(1).to_string(index=False))
# gem-dimethyl regio: in Ap/Dp, which orientation wins (Me2-carbon bonded to terminus a or b) and by how much
print('\n-- gem-dimethyl orientation (Ap, Dp): ΔE‡(a − b) median over R1, frac a preferred')
for sk in ['Ap','Dp']:
    for dp in sorted(d.dipole.unique()):
        try:
            a=bestO.xs((sk,'a'),level=['skeleton','orient']).loc[dp]; b=bestO.xs((sk,'b'),level=['skeleton','orient']).loc[dp]
        except KeyError: continue
        cm=a.index.intersection(b.index)
        if len(cm)<5: continue
        dd=a.loc[cm,'barrier_kcal']-b.loc[cm,'barrier_kcal']; ds=(a.loc[cm,'strain']-b.loc[cm,'strain']); de=(a.loc[cm,'Eint']-b.loc[cm,'Eint'])
        print(f'{sk:3s} {dp:16s} n={len(cm):3d}  ΔE‡(a−b) median {dd.median():+.1f}  a preferred {(dd<0).mean():.2f}  | sign agree strain {(np.sign(ds)==np.sign(dd)).mean():.2f}  Eint {(np.sign(de)==np.sign(dd)).mean():.2f}')
