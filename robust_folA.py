"""Noise-aware reanalysis of the Papkou et al. (2023) folA landscape on the nucleotide graph (4^9).
Fitness m and replicate-based standard errors m_se from the authors' Zenodo deposit (record 13837982).
- unobserved variants removed; uphill steps must exceed z combined SEs
- order-k null models fitted in a codon basis (64^3): k=1 additive, k=2 pairwise across codons
- null: model + Gaussian noise with the measured SEs (scaled by c = 0.5, 1, 2)
- depth thinning of the six-replicate read counts, fitness re-estimated from log count ratios
- mean imputation + tie-break, as in the naive GB1 pipeline"""
import numpy as np, pandas as pd, json, robust
from common import nb_slice

from robust_folA_data import NT, L, S, WT, FUNC, idx, F, SE, mask, F0, SE0, target, models, load_counts
rng = np.random.default_rng(2026)
out = dict(observed=int(mask.sum()), target=target, func_thr=FUNC)

# edge resolution
tot = r1 = r2 = floor = 0
for a in range(L):
    for b in range(S):
        both = mask & nb_slice(mask, a, b)
        up = (nb_slice(F0, a, b) > F0) & both
        g = (nb_slice(F0, a, b) - F0)[up]; s = np.sqrt(SE0**2 + nb_slice(SE0, a, b)**2)[up]
        tot += up.sum(); r1 += (g > s).sum(); r2 += (g > 2*s).sum()
        floor += ((F0 < FUNC) & (nb_slice(F0, a, b) < FUNC))[up].sum()
out['edges'] = dict(n=int(tot), resolved_1se=float(r1/tot), resolved_2se=float(r2/tot), both_nonfunctional=float(floor/tot))
print('edges', out['edges'], flush=True)

# data at three thresholds
out['data'] = {z: robust.summarize(F0, SE0, mask, target, z, FUNC) for z in (0, 1, 2)}
print('data', out['data'], flush=True)

# naive pipeline: impute to mean, tie-break, all genotypes present
perm = (rng.permutation(F.size)/F.size).reshape(F.shape)
Fi = np.where(mask, F, F[mask].mean()) + 1e-9*perm; full = np.ones_like(mask)
p = robust.basins(Fi, full, robust.edges(Fi, np.zeros_like(Fi), full, 0))
pk = np.unique(p)
out['imputed_pipeline'] = dict(peaks=int(len(pk)), imputed_peaks=int((~mask.ravel()[pk]).sum()),
                               imputed_value=float(F[mask].mean()), median=float(np.median(F[mask])))
print('imputed pipeline', out['imputed_pipeline'], flush=True)

out['models'] = {k: robust.summarize(M, SE0, mask, target, 0, FUNC) for k, M in models.items()}
print('models', out['models'], flush=True)

# null: model + measured-scale Gaussian noise; bootstrap of the data with the same noise
out['null'] = {}
for k in (1, 2):
    out['null'][k] = {}
    for c in (0.5, 1, 2):
        out['null'][k][c] = [robust.summarize(models[k] + c*SE0*rng.normal(size=F.shape), SE0, mask, target, 0, FUNC)
                             for _ in range(50)]
        print('null k=%d c=%g' % (k, c), np.median([r['peaks'] for r in out['null'][k][c]]), flush=True)
out['bootstrap'] = [robust.summarize(F0 + SE0*rng.normal(size=F.shape), SE0, mask, target, 0, FUNC) for _ in range(50)]

# depth thinning of the replicate counts
A, B, estimate = load_counts()
G, _ = estimate(A, B); both = mask & ~np.isnan(G)
out['count_estimator_corr'] = float(np.corrcoef(G[both], F[both])[0, 1])
print('count estimator vs published m: r =', out['count_estimator_corr'], flush=True)
rows = []
for q in [0.05, 0.1, 0.25, 0.5, 1.0]:
    reps = []
    for _ in range(5 if q < 1 else 1):
        Aq = rng.binomial(A.astype(int), q) if q < 1 else A
        Bq = rng.binomial(B.astype(int), q) if q < 1 else B
        G, E = estimate(Aq, Bq); m = ~np.isnan(G)
        tq = int(np.argmax(np.where(m, G, -np.inf)))
        reps.append(robust.summarize(np.where(m, G, 0), np.where(m, E, 0), m, target if m.flat[target] else tq, 0, FUNC))
    rows.append(dict(q=q, **{key: float(np.mean([r[key] for r in reps])) for key in reps[0]}))
    print('thin', rows[-1], flush=True)
out['thin'] = rows
json.dump(out, open('robust_folA_results.json', 'w'), indent=1)
print('saved robust_folA_results.json')
