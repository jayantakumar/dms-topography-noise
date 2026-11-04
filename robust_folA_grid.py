"""folA low-order nulls on a noise-scale grid: codon-additive and codon-pairwise models plus Gaussian noise of
c times the published replicate SE, 200 replicates per (model, c). Every replicate's statistics are saved so that
rank p-values and the joint (Mahalanobis) test in joint_tests.py can be computed from this file."""
import numpy as np, json, robust
import robust_folA_data as D

SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5); REPS = 200; KEYS = ('support', 'peaks', 'func_peaks', 'P')
rng = np.random.default_rng(11)
data = robust.summarize(D.F0, D.SE0, D.mask, D.target, 0, D.FUNC)
out = dict(data={k: data[k] for k in KEYS}, scales=list(SCALES), reps=REPS, null={})
for k in (1, 2):
    out['null'][k] = {}
    for c in SCALES:
        reps = [robust.summarize(D.models[k] + c*D.SE0*rng.normal(size=D.F0.shape), D.SE0, D.mask, D.target, 0, D.FUNC)
                for _ in range(REPS)]
        out['null'][k][c] = {key: [r[key] for r in reps] for key in KEYS}
        print(k, c, {key: round(float(np.median(out['null'][k][c][key])), 4) for key in KEYS}, flush=True)
json.dump(out, open('robust_folA_grid_results.json', 'w'), indent=1)
print('saved robust_folA_grid_results.json')
