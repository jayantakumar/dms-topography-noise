"""folA (Papkou et al. 2023) fitness, replicate SEs and codon-basis low-order fits on the 4^9 nucleotide graph."""
import numpy as np, pandas as pd, robust

NT = "ACGT"; L = 9; S = 4; WT = "GCCGATCTC"; FUNC = -0.507774   # Papkou functional threshold
def idx(sv): return tuple(NT.index(c) for c in sv)

d = pd.read_csv('data/folA_raw/fitness_data_wt.csv')
F = np.full((S,)*L, np.nan); SE = np.full((S,)*L, np.nan)
for sv, m, se in zip(d.SV, d.m, d.m_se): F[idx(sv)] = m; SE[idx(sv)] = se
mask = ~np.isnan(F); SE[idx(WT)] = np.nanmin(SE)
F0 = np.where(mask, F, 0.0); SE0 = np.where(mask, SE, 0.0)
target = int(np.argmax(np.where(mask, F, -np.inf)))

# codon-basis order-k models
C = F0.reshape(64, 64, 64); Cm = mask.reshape(64, 64, 64); order = robust.order_index(3, 64)
models = {k: robust.fit_order(C, Cm, k, order).reshape((S,)*L) for k in (1, 2)}

def load_counts():
    """Replicate read counts per variant (rows aligned across before/after) and a fitness estimator:
    estimate(A, B) -> (fitness tensor, SE tensor) from the mean over replicates of the
    wild-type-normalized log ratio; with one replicate the SE is the Poisson delta-method error."""
    def amplicon_counts(path):
        t = pd.read_csv(path, index_col=0)
        sv = [s[s.find('AACCTGCCT')+9:][:9] for s in t.index]
        return t.groupby(sv).sum()
    bef = amplicon_counts('data/folA_raw/sequence_reads_counts_before_selection.csv')
    aft = amplicon_counts('data/folA_raw/sequence_reads_counts_after_selection.csv')
    common_sv = bef.index.intersection(aft.index)
    B = bef.loc[common_sv].values.astype(float); A = aft.loc[common_sv].values.astype(float)
    cells = tuple(np.array([idx(s) for s in common_sv]).T)
    wt_row = list(common_sv).index(WT)
    def estimate(A, B):
        lr = np.log((A + 0.5)/(B + 0.5)); lr -= lr[wt_row]
        G = np.full((S,)*L, np.nan); E = np.full((S,)*L, np.nan)
        G[cells] = lr.mean(1)
        if lr.shape[1] > 1:
            E[cells] = lr.std(1, ddof=1)/np.sqrt(lr.shape[1])
        else:
            E[cells] = np.sqrt(1/(A[:, 0] + 0.5) + 1/(B[:, 0] + 0.5))
        return G, E
    return A, B, estimate
