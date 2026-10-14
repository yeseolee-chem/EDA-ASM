"""figs.py — run_all.py 출력 폴더에서 그림 4장을 만든다. python figs.py OUTDIR"""
import sys, os, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
D=sys.argv[1] if len(sys.argv)>1 else '.'
FIG=os.path.join(D,'fig'); os.makedirs(FIG,exist_ok=True)
SURF='#fcfcfb'; T1='#0b0b0b'; T2='#52514e'; GRID='#e4e3df'
S=['#2a78d6','#eb6834','#1baf7a','#eda100']
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.size':9,'axes.edgecolor':'#bdbcb6',
 'axes.labelcolor':T2,'xtick.color':T2,'ytick.color':T2,'text.color':T1,'axes.titlesize':10,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
order=['d1','d2','elst','Pauli','OI','disp','CPCM','CDS','bsse']
# ---------- Fig 1 budget
B=pd.read_csv(os.path.join(D,'budget.csv')).set_index('set')
sets=[('ALL pairs','D1 substituent pairs (ΔΔ, n=7,632)'),('D1 absolute (between all TS)','D1 between reactions (n=9,231)'),('D0 absolute (between all TS)','D0 between reactions (n=5,262)')]
fig,ax=plt.subplots(figsize=(7.2,4.2))
y=np.arange(len(order)); h=0.26
for j,(k,lab) in enumerate(sets):
    v=B.loc[k,order].values.astype(float)
    ax.barh(y+(j-1)*h,v,height=h-0.03,color=S[j],label=lab,zorder=3)
    if j==0:
        for yi,vi in zip(y,v):
            if abs(vi)>0.3: ax.text(vi+(0.05 if vi>0 else -0.05),yi-h,f'{vi:+.2f}',va='center',ha='left' if vi>0 else 'right',fontsize=8,color=T1)
ax.axvline(0,color='#8c8b85',lw=1,zorder=2); ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
ax.grid(axis='x',color=GRID,lw=0.8,zorder=0); ax.set_xlabel('share of Var(ΔE‡)  =  Cov(channel, ΔE‡) / Var(ΔE‡)   (shares sum to 1)')
ax.set_title('Who carries the barrier change? Covariance budget of ΔE‡')
ax.legend(frameon=False,loc='lower right',fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig1_budget.png'),dpi=200); plt.close(fig)
# ---------- Fig 2 low-rank
L=pd.read_csv(os.path.join(D,'lowrank.csv'))
models=[('R2_subst_only','scalar\nper X'),('R2_additive','X + context\n(additive)'),('R2_rank1','rank 1\n(σ·ρ)'),('R2_rank2','rank 2'),('R2_rank3','rank 3')]
chs=['ΔE‡','Pauli','OI','disp']
fig,axs=plt.subplots(1,2,figsize=(8.6,3.6),sharey=True)
for ax,side,title in zip(axs,['dipole','dipolarophile'],['substituent on dipole (74 contexts × 20 X)','substituent on dipolarophile (31 contexts × 20 X)']):
    q=L[(L.side==side)&(L.set=='direct')].set_index('ch'); ends=[]
    for j,c in enumerate(chs):
        v=q.loc[c,[m for m,_ in models]].values.astype(float)
        ax.plot(range(len(models)),v,'-o',color=S[j],lw=2,ms=5,label=c,zorder=3)
        ends.append([v[-1],c])
    ends.sort(key=lambda e:e[0]); ys=[e[0] for e in ends]
    for i in range(1,len(ys)):
        if ys[i]-ys[i-1]<0.045: ys[i]=ys[i-1]+0.045
    for (v,c),yy in zip(ends,ys): ax.text(len(models)-1+0.12,yy,c,color=T1,fontsize=8,va='center')
    ax.set_xticks(range(len(models))); ax.set_xticklabels([l for _,l in models],fontsize=7)
    ax.axhline(0,color='#8c8b85',lw=1); ax.grid(axis='y',color=GRID,lw=0.8,zorder=0); ax.set_title(title,fontsize=9)
    ax.set_xlim(-0.3,len(models)-0.2)
axs[0].set_ylabel('5-fold CV R² on held-out (context, X) cells')
axs[0].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('How many substituent parameters are needed? (direct substituents, ΔΔ vs H parent)',fontweight='bold',fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig2_lowrank.png'),dpi=200); plt.close(fig)
# ---------- Fig 3 selectivity
Sx=pd.read_csv(os.path.join(D,'selectivity.csv')).set_index('set')
items=[('acc_d1','d1'),('acc_d2','d2'),('acc_strain','d1+d2'),('acc_elst','elst'),('acc_Pauli','Pauli'),('acc_OI','OI'),('acc_Eint','E_int (6ch+bsse)'),('acc_earlierTS','earlier TS (larger ζ)')]
fig,axs=plt.subplots(1,2,figsize=(8.6,3.4),sharey=True)
for ax,k,title in zip(axs,['regio a−b (all)','exo−endo (all)'],[f"regioselectivity (n={int(Sx.loc['regio a−b (all)','n'])} pairs)",f"exo/endo selectivity (n={int(Sx.loc['exo−endo (all)','n'])} pairs)"]):
    v=Sx.loc[k,[a for a,_ in items]].values.astype(float)
    ax.bar(range(len(items)),v,color=S[0],width=0.7,zorder=3)
    for i,vi in enumerate(v): ax.text(i,vi+0.02,f'{vi:.2f}',ha='center',fontsize=7.5,color=T1)
    ax.axhline(0.5,color='#8c8b85',lw=1,ls='--',zorder=2); ax.text(-0.45,0.515,'coin flip',fontsize=7,color=T2,ha='left',va='bottom')
    ax.set_xticks(range(len(items))); ax.set_xticklabels([l for _,l in items],rotation=35,ha='right',fontsize=7.5)
    ax.set_ylim(0,1.08); ax.grid(axis='y',color=GRID,lw=0.8,zorder=0); ax.set_title(title,fontsize=9)
axs[0].set_ylabel('fraction of pairs where the channel\nfavours the lower-barrier TS')
fig.suptitle('Which channel picks the winning TS? (single point at each TS; dashed line = 0.5)',fontweight='bold',fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig3_selectivity.png'),dpi=200); plt.close(fig)
# ---------- Fig 4 fingerprint heatmap
FP=pd.read_csv(os.path.join(D,'fingerprint_direct.csv'))
cols=['d1','d2','elst','Pauli','OI','disp','CPCM','CDS','ΔE‡']
cmap=LinearSegmentedColormap.from_list('div',['#104281','#3987e5','#f0efec','#e34948','#9e2423'])
fig,axs=plt.subplots(1,2,figsize=(9.4,5.6))
for ax,side,title in zip(axs,['dipolarophile','dipole'],['X on dipolarophile','X on dipole']):
    q=FP[FP.side==side].set_index('X')[cols].sort_values('ΔE‡')
    Z=q/q.abs().max()  # per-column scale
    ax.imshow(Z.values,cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            val=q.values[i,j]; ax.text(j,i,f'{val:.1f}',ha='center',va='center',fontsize=6.5,color=T1 if abs(Z.values[i,j])<0.6 else 'white')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,fontsize=8); ax.set_yticks(range(q.shape[0])); ax.set_yticklabels(q.index,fontsize=8)
    ax.set_title(title,fontsize=9); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle('Median ΔΔ (kcal/mol) vs H parent, direct substituents (colour scaled per column; blue = lowers, red = raises)',fontweight='bold',fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig4_fingerprint.png'),dpi=200); plt.close(fig)
print('ok')
