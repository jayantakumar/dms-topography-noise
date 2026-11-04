import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15,'axes.titlesize':15,'axes.labelsize':14,
                     'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':11.5})
A = json.load(open('robust_folA_results.json')); G = json.load(open('robust_results.json'))
GD = json.load(open('robust_depth_results.json')); GR = json.load(open('robust_folA_grid_results.json'))
C = {'1':'#DD8452', '2':'#4C72B0', 'data':'#C44E52', 'gb1':'#555555'}
cs = GR['scales']
fig, ax = plt.subplots(2, 2, figsize=(12, 10.5))

# A: thinning, folA vs GB1 (peaks relative to full depth)
for src, rows, col, name in [('folA', A['thin'], C['data'], 'folA'), ('GB1', GD['thin'], C['gb1'], 'GB1')]:
    q = [r['q'] for r in rows]; p0 = rows[-1]['peaks']; P0 = rows[-1]['P']
    ax[0,0].semilogx(q, [r['peaks']/p0 for r in rows], 'o-', color=col, label=f'{name}: peaks')
    ax[0,0].semilogx(q, [r['P']/P0 for r in rows], 's--', color=col, label=f'{name}: P')
ax[0,0].axhline(1, color='k', lw=0.6)
ax[0,0].set_xlabel('fraction of sequencing depth retained'); ax[0,0].set_ylabel('value relative to full depth')
ax[0,0].legend(); ax[0,0].set_title('A. Depth convergence: folA vs GB1')

# B, C: null versus noise scale
for panel, key, ylab, title in [(ax[0,1], 'func_peaks', 'functional peaks', 'B. folA functional peaks vs low-order nulls'),
                                (ax[1,0], 'P', 'basin concentration P', 'C. folA basin concentration vs low-order nulls')]:
    for k, name in [('1','additive (codons)'), ('2','pairwise (codons)')]:
        reps = [GR['null'][k][str(float(c))][key] for c in cs]
        m = [np.median(v) for v in reps]; lo = [np.quantile(v, .025) for v in reps]; hi = [np.quantile(v, .975) for v in reps]
        panel.plot(cs, m, 'o-', color=C[k], label=f'{name} + noise'); panel.fill_between(cs, lo, hi, color=C[k], alpha=0.2)
    panel.axhline(A['data']['0'][key], color=C['data'], ls='--', label='folA data')
    panel.set_xscale('log', base=2); panel.set_xticks([0.5, 1, 2]); panel.set_xticklabels(['0.5×','1×','2×'])
    panel.set_xlabel('noise scale (× measured replicate SE)'); panel.set_ylabel(ylab); panel.set_yscale('log')
    panel.legend(); panel.set_title(title)

# D: support vs significance threshold, both landscapes
zs = [0, 1, 2]
ax[1,1].plot(zs, [100*A['data'][str(z)]['support'] for z in zs], 'o-', color=C['data'], label='folA')
ax[1,1].plot(zs, [100*G['models']['4'][str(z)]['support'] for z in zs], 's-', color=C['gb1'], label='GB1')
ax[1,1].set_xticks(zs); ax[1,1].set_xlabel('each step must exceed z standard errors')
ax[1,1].set_ylabel('% genotypes that can reach $x^*$'); ax[1,1].set_ylim(0, 102)
ax[1,1].legend(); ax[1,1].set_title('D. Support depends on the evidence required')
plt.tight_layout(); plt.savefig('fig_folA.png', dpi=200); print('saved')
