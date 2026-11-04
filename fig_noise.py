import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15,'axes.titlesize':15,'axes.labelsize':14,
                     'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':11.5})
R = json.load(open('robust_results.json')); D = json.load(open('robust_depth_results.json'))
C = {'2':'#4C72B0', '3':'#55A868', 'data':'#C44E52'}
fig, ax = plt.subplots(2, 2, figsize=(12, 10.5))

# A: support by model order (noise-free fits, z = 0) and data at three significance thresholds
ks = ['1','2','3','4']; lab = ['additive','pairwise','three-way','full (data)']
ax[0,0].plot(range(4), [100*R['models'][k]['0']['support'] for k in ks], 'o-', color='#333333', label='any increase (z = 0)')
for z, mk in [('1','s'), ('2','^')]:
    ax[0,0].plot([3], [100*R['models']['4'][z]['support']], mk, ms=8, color=C['data'], label=f'data, step must exceed {z} SE')
ax[0,0].set_xticks(range(4)); ax[0,0].set_xticklabels(lab)
ax[0,0].set_ylabel('% genotypes that can reach $x^*$'); ax[0,0].legend(loc='lower right')
ax[0,0].set_title('A. Support is dominated by low-order terms')

# B: thinning the real counts
th = D['thin']; q = [r['q'] for r in th]
ax[0,1].loglog(q, [r['peaks'] for r in th], 'o-', color=C['data'], label='peaks (all)')
ax[0,1].loglog(q, [r['func_peaks'] for r in th], 's--', color=C['data'], label=r'peaks with $f\geq0.1$')
ax[0,1].set_xlabel('fraction of sequencing depth retained'); ax[0,1].set_ylabel('number of peaks')
b2 = ax[0,1].twinx(); b2.semilogx(q, [r['P'] for r in th], '^:', color='#8172B2', label='P')
b2.set_ylabel('basin concentration P', color='#8172B2'); b2.tick_params(axis='y', labelcolor='#8172B2')
h1, l1 = ax[0,1].get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
ax[0,1].legend(h1+h2, l1+l2, loc='upper right'); ax[0,1].set_title('B. GB1 peaks and P depend on depth')

# C: low-order models simulated at increasing depth
for k, name in [('2','pairwise'), ('3','three-way')]:
    rows = [r for r in D['sim'][k] if r['m'] != 'inf']; inf = [r for r in D['sim'][k] if r['m'] == 'inf'][0]
    ax[1,0].loglog([r['m'] for r in rows], [r['peaks'] for r in rows], 'o-', color=C[k], label=f'{name} model + counting noise')
    ax[1,0].axhline(inf['peaks'], ls=':', color=C[k], lw=1)
ax[1,0].plot([1], [R['models']['4']['0']['peaks']], '*', ms=14, color=C['data'], label='GB1 data (depth ×1)')
ax[1,0].set_xlabel('sequencing depth (× actual)'); ax[1,0].set_ylabel('number of peaks')
ax[1,0].legend(loc='upper right'); ax[1,0].set_title('C. Counting noise inflates peak counts')

# D: null distributions of P
bins = np.linspace(0.04, 0.32, 29)
for key, name, col in [('2','pairwise + noise',C['2']), ('3','three-way + noise',C['3'])]:
    ax[1,1].hist([r['P'] for r in R['null'][key]['0']], bins=bins, alpha=0.55, color=col, label=name)
ax[1,1].hist([r['P'] for r in R['bootstrap']['0']], bins=bins, histtype='step', lw=2, color=C['data'], label='GB1 bootstrap (spread only)')
ax[1,1].axvline(R['models']['4']['0']['P'], color=C['data'], ls='--', label='GB1 point estimate')
ax[1,1].set_xlabel('basin concentration P'); ax[1,1].set_ylabel('replicates')
ax[1,1].legend(); ax[1,1].set_title('D. Data P lies inside the low-order nulls')
plt.tight_layout(); plt.savefig('fig_noise.png', dpi=200); print('saved')
