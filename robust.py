"""Noise-aware accessibility: missing genotypes removed from the graph, uphill steps
must exceed z combined standard errors, peaks reported above a functional threshold."""
import numpy as np
from itertools import product
from common import nb_slice

def edge_ok(F, SE, a, b, z):
    """Boolean array: allele b at site a is significantly fitter than the current genotype."""
    thr = z*np.sqrt(SE**2 + nb_slice(SE, a, b)**2)
    with np.errstate(invalid='ignore'):
        return nb_slice(F, a, b) - F > thr

def edges(F, SE, mask, z):
    """Precomputed significant-uphill masks [(a, b, ok)], restricted to observed endpoints."""
    return [(a, b, edge_ok(F, SE, a, b, z) & mask & nb_slice(mask, a, b))
            for a in range(F.ndim) for b in range(F.shape[0])]

def reach(F, mask, target, E):
    """Genotypes in mask with a significantly-uphill path (within mask) to target."""
    R = np.zeros(F.shape, bool); R.flat[target] = True
    front = R.copy()
    while front.any():
        new = np.zeros(F.shape, bool)
        for a, b, ok in E:
            new |= ok & nb_slice(front, a, b)
        new &= ~R
        R |= new; front = new
    return R

def basins(F, mask, E):
    """Steepest significant ascent; returns terminal peak per genotype (flat), -1 outside mask."""
    n = F.size
    idx = np.arange(n).reshape(F.shape)
    best = np.where(mask, F, -np.inf); ptr = idx.copy()
    for a, b, ok in E:
        Fb = nb_slice(F, a, b)
        better = ok & (Fb > best)
        best = np.where(better, Fb, best); ptr = np.where(better, nb_slice(idx, a, b), ptr)
    p = ptr.ravel()
    while True:
        q = p[p]
        if np.array_equal(q, p): break
        p = q
    return np.where(mask.ravel(), p, -1)

def summarize(F, SE, mask, target, z, func_thr):
    E = edges(F, SE, mask, z)
    p = basins(F, mask, E); m = mask.ravel()
    peaks, sizes = np.unique(p[m], return_counts=True)
    fv = F.ravel()[peaks]; N = m.sum()
    return dict(support=float(reach(F, mask, target, E)[mask].mean()),
                peaks=int(len(peaks)), func_peaks=int((fv >= func_thr).sum()),
                P=float(np.sum((sizes/N)**2)),
                P_func=float(np.sum((sizes[fv >= func_thr]/N)**2)))

def order_index(L, S):
    return (np.array(list(product(range(S), repeat=L))) != 0).sum(1).reshape((S,)*L)

def fit_order(Y, mask, k, order, iters=1000, tol=1e-10):
    """Least-squares order<=k Fourier model fitted to observed cells only (EM projection)."""
    keep = order <= k
    X = np.where(mask, Y, Y[mask].mean())
    for _ in range(iters):
        R = np.real(np.fft.ifftn(np.where(keep, np.fft.fftn(X), 0)))
        Xn = np.where(mask, Y, R)
        if np.max(np.abs(Xn - X)) < tol: break
        X = Xn
    return R
