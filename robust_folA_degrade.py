"""Controlled degradation of folA towards GB1's measurement regime.
Conditions: six replicates (reference) -> one replicate -> one replicate thinned to GB1's median read depth.
In each condition fitness is re-estimated from counts, a codon-pairwise model is fitted to that estimate,
and the data are compared with that model plus Gaussian noise of the condition's own standard errors.
If measurement regime drives identifiability, the degraded folA should become pairwise-plus-noise compatible."""
import numpy as np, json, robust
from robust_folA_data import F, mask, target, FUNC, load_counts, S, L

GB1_SEL, GB1_INP = 4.0, 249.0             # GB1 median selected / input reads per variant
NNULL = 20
rng = np.random.default_rng(31)
A, B, estimate = load_counts()
order = robust.order_index(3, 64)

# functional threshold on the count-estimator scale, by linear map from the published m
G6, _ = estimate(A, B); ok = mask & ~np.isnan(G6)
slope, icpt = np.polyfit(F[ok], G6[ok], 1); thr = icpt + slope*FUNC

def analyse(G, E):
    m = ~np.isnan(G); G0 = np.where(m, G, 0.0); E0 = np.where(m, E, 0.0)
    tgt = target if m.flat[target] else int(np.argmax(np.where(m, G, -np.inf)))
    data = robust.summarize(G0, E0, m, tgt, 0, thr)
    M = robust.fit_order(G0.reshape(64, 64, 64), m.reshape(64, 64, 64), 2, order).reshape((S,)*L)
    null = [robust.summarize(M + E0*rng.normal(size=G0.shape), E0, m, tgt, 0, thr) for _ in range(NNULL)]
    band = {k: [float(np.quantile([r[k] for r in null], q)) for q in (.025, .5, .975)]
            for k in ('support', 'peaks', 'func_peaks', 'P')}
    inside = {k: bool(band[k][0] <= data[k] <= band[k][2]) for k in band}
    return dict(data=data, null=band, inside=inside, median_selected=float(np.median(A_used)), median_input=float(np.median(B_used)))

out = dict(threshold=float(thr), conditions={})
A_used, B_used = A, B
out['conditions']['six replicates'] = analyse(*estimate(A, B))
print('six replicates', out['conditions']['six replicates'], flush=True)
for r in range(A.shape[1]):
    a, b = A[:, [r]], B[:, [r]]
    A_used, B_used = a, b
    out['conditions'][f'replicate {r+1}'] = analyse(*estimate(a, b))
    qa = min(1.0, GB1_SEL/np.median(a)); qb = min(1.0, GB1_INP/np.median(b))
    a2 = rng.binomial(a.astype(int), qa).astype(float); b2 = rng.binomial(b.astype(int), qb).astype(float)
    A_used, B_used = a2, b2
    out['conditions'][f'replicate {r+1}, GB1 depth'] = analyse(*estimate(a2, b2))
    for name in (f'replicate {r+1}', f'replicate {r+1}, GB1 depth'):
        c = out['conditions'][name]
        print(name, {k: (round(c['data'][k], 4), [round(x, 4) for x in c['null'][k]]) for k in c['null']}, flush=True)
json.dump(out, open('robust_folA_degrade_results.json', 'w'), indent=1)
print('saved robust_folA_degrade_results.json')
