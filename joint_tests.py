"""Joint and marginal tests of the data against each low-order null.
Statistics: support, log peaks, log functional peaks (folA only), log P. The joint test is the Mahalanobis
distance of the data from the null replicates; its p-value is the rank of the data's distance among the
replicates' distances, (1 + #{d_rep >= d_data}) / (1 + n). Marginal p-values are two-sided ranks.
GB1 omits functional peaks because its nulls have a higher fitness floor than the data (see paper)."""
import numpy as np, json

def vec(d, keys):
    return np.array([d['support'] if k == 'support' else np.log(d[k]) for k in keys])

def test(data, reps, keys):
    X = np.array([vec(r, keys) for r in reps]); x = vec(data, keys)
    mu = X.mean(0); Ci = np.linalg.pinv(np.cov(X, rowvar=False))
    d2 = lambda v: float((v - mu) @ Ci @ (v - mu))
    dd = np.array([d2(v) for v in X]); n = len(X)
    marg = {}
    for j, k in enumerate(keys):
        lo = (1 + np.sum(X[:, j] <= x[j]))/(1 + n); hi = (1 + np.sum(X[:, j] >= x[j]))/(1 + n)
        marg[k] = float(min(1.0, 2*min(lo, hi)))
    return dict(n=n, d2=d2(x), p_joint=float((1 + np.sum(dd >= d2(x)))/(1 + n)), p_marginal=marg)

out = {'GB1': {}, 'folA': {}}
G = json.load(open('robust_results.json')); gkeys = ('support', 'peaks', 'P')
for k, name in (('2', 'pairwise'), ('3', 'three-way')):
    out['GB1'][name] = test(G['models']['4']['0'], G['null'][k]['0'], gkeys)
    print('GB1', name, out['GB1'][name])
A = json.load(open('robust_folA_grid_results.json')); fkeys = ('support', 'peaks', 'func_peaks', 'P')
for k, name in (('1', 'additive'), ('2', 'pairwise')):
    out['folA'][name] = {}
    for c, null in A['null'][k].items():
        reps = [dict(zip(null, vals)) for vals in zip(*null.values())]
        out['folA'][name][c] = test(A['data'], reps, fkeys)
        r = out['folA'][name][c]
        print('folA', name, c, 'p_joint', round(r['p_joint'], 4), {q: round(v, 3) for q, v in r['p_marginal'].items()})
json.dump(out, open('joint_tests_results.json', 'w'), indent=1)
print('saved joint_tests_results.json')
