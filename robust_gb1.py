"""Noise- and imputation-aware reanalysis of GB1.
- unobserved genotypes are removed from the graph (no imputation)
- an uphill step must exceed z combined Poisson standard errors (z = 0, 1, 2)
- order-k models are least-squares fits to observed genotypes only
- count-level parametric bootstrap of the data
- null test: counts simulated from the pairwise / three-way model, same pipeline"""
import numpy as np, json, time, robust

from robust_gb1_data import sel, inp, mask, norm, fitness, stderr, F, SE, target, order
ZS = [0, 1, 2]; FUNC = 0.1; B = 50; NNULL = 200
rng = np.random.default_rng(2026)
out = dict(target=target, func_thr=FUNC, models={}, bootstrap={}, null={})

models = {k: robust.fit_order(F, mask, k, order) for k in (1, 2, 3)}
models[4] = F
for k, M in models.items():
    out['models'][k] = {z: robust.summarize(M, SE, mask, target, z, FUNC) for z in ZS}
    print('model k<=%d' % k, out['models'][k], flush=True)

def resample(lam_sel):
    s = rng.poisson(lam_sel).astype(float)
    i = np.maximum(rng.poisson(inp), 1).astype(float)
    return fitness(s, i), stderr(s, i)

t = time.time()
for z in ZS:
    reps = []
    for _ in range(B):
        Fb, SEb = resample(sel)
        reps.append(robust.summarize(Fb, SEb, mask, target, z, FUNC))
    out['bootstrap'][z] = reps
    print('bootstrap z=%d done (%.0fs)' % (z, time.time() - t), flush=True)

for k in (2, 3):
    lam = np.clip(models[k], 0, None)*inp/norm
    out['null'][k] = {}
    for z in ZS:
        reps = []
        for _ in range(NNULL):
            Fn, SEn = resample(lam)
            reps.append(robust.summarize(Fn, SEn, mask, target, z, FUNC))
        out['null'][k][z] = reps
    print('null k<=%d done (%.0fs)' % (k, time.time() - t), flush=True)

json.dump(out, open('robust_results.json', 'w'), indent=1)
print('saved robust_results.json')
