"""Sequencing-depth dose-response.
(1) counts simulated from the pairwise / three-way model at depth multiplier m (inf = noise-free model)
(2) the real GB1 counts binomially thinned to fraction q of their depth"""
import numpy as np, json, robust
from robust_gb1_data import sel, inp, mask, norm, fitness, stderr, target, order

FUNC = 0.1; REPS = 10
rng = np.random.default_rng(7)
F = fitness(sel, inp)
out = dict(sim={}, thin={})
for k in (2, 3):
    M = robust.fit_order(F, mask, k, order)
    rows = [dict(m='inf', **robust.summarize(M, np.zeros_like(M), mask, target, 0, FUNC))]
    for m in [0.1, 0.3, 1, 3, 10, 100]:
        reps = []
        for _ in range(REPS):
            i = np.maximum(rng.poisson(inp*m), 1).astype(float)
            s = rng.poisson(np.clip(M, 0, None)*i/norm).astype(float)
            reps.append(robust.summarize(fitness(s, i), stderr(s, i), mask, target, 0, FUNC))
        rows.append(dict(m=m, **{key: float(np.mean([r[key] for r in reps])) for key in reps[0]}))
        print('model k<=%d depth x%g' % (k, m), rows[-1], flush=True)
    out['sim'][k] = rows
rows = []
for q in [0.05, 0.1, 0.25, 0.5, 1.0]:
    reps = []
    for _ in range(REPS if q < 1 else 1):
        s = rng.binomial(sel.astype(int), q).astype(float) if q < 1 else sel
        i = np.maximum(rng.binomial(inp.astype(int), q), 1).astype(float) if q < 1 else inp
        reps.append(robust.summarize(fitness(s, i), stderr(s, i), mask, target, 0, FUNC))
    rows.append(dict(q=q, **{key: float(np.mean([r[key] for r in reps])) for key in reps[0]}))
    print('data thinned to %g' % q, rows[-1], flush=True)
out['thin'] = rows
json.dump(out, open('robust_depth_results.json', 'w'), indent=1)
print('saved robust_depth_results.json')
