"""Shared machinery: multiallelic WHT, directed reachability, basin partition."""
import numpy as np
from itertools import product

def wht(T):
    A = T.astype(complex)
    for ax in range(T.ndim):
        A = np.fft.fft(A, axis=ax)
    return A / np.sqrt(T.size)

def iwht(P):
    A = P.astype(complex)
    for ax in range(P.ndim):
        A = np.fft.ifft(A, axis=ax)
    return np.real(A * np.sqrt(P.size))

def order_mask(L, S):
    keys = np.stack(np.meshgrid(*[np.arange(S)]*L, indexing='ij'), axis=-1).reshape(-1, L)
    return (keys != 0).sum(1)

def variance_components(P):
    L = P.ndim; S = P.shape[0]
    mag = np.abs(P).ravel()**2
    o = order_mask(L, S)
    tot = mag.sum()
    return np.array([mag[o == k].sum()/tot for k in range(L+1)])

def nb_slice(F, a, b):
    """F with site a fixed to allele b, kept aligned with F (axis a has length 1)."""
    return np.expand_dims(np.take(F, b, axis=a), a)

def peak_density(F):
    """Fraction of genotypes with no strictly fitter single-mutant neighbour."""
    higher = np.zeros(F.shape, dtype=bool)
    for a in range(F.ndim):
        for b in range(F.shape[0]):
            higher |= nb_slice(F, a, b) > F
    return float((~higher).mean())

def reachable_set(F, nbatch=16, sources=None):
    """Boolean: node can reach a source via strictly increasing steps.
    sources: flat indices (default: the global max)."""
    shape = F.shape; L = len(shape); S = shape[0]; n = F.size
    R = np.zeros(shape, dtype=bool)
    if sources is None:
        R.flat[int(np.argmax(F))] = True
    else:
        R.flat[np.asarray(sources)] = True
    Fv = F.ravel(); order = np.argsort(-Fv, kind='stable')
    batch = max(1, n//nbatch)
    for start in range(0, n, batch):
        blk = order[start:start+batch]
        anyf = np.zeros(shape, dtype=bool)
        for a in range(L):
            for b in range(S):
                anyf |= (nb_slice(F, a, b) > F) & nb_slice(R, a, b)
        R.ravel()[blk] = anyf.ravel()[blk]
    return R

def steepest_pointer(F):
    """Flat index of the steepest uphill neighbour, or self if local max."""
    shape = F.shape; L = len(shape); S = shape[0]; n = F.size
    idx = np.arange(n).reshape(shape)
    best = F.copy(); ptr = idx.copy()
    for a in range(L):
        for b in range(S):
            Fb = nb_slice(F, a, b)
            better = Fb > best
            best = np.where(better, Fb, best)
            ptr = np.where(better, nb_slice(idx, a, b), ptr)
    return ptr.ravel()

def basins(F, max_iter=64):
    p = steepest_pointer(F)
    for _ in range(max_iter):
        q = p[p]
        if np.array_equal(q, p): break
        p = q
    sizes = np.bincount(p, minlength=F.size)
    return p, sizes[sizes > 0]

def predictability(F):
    """P(two random starts end at same local max) = sum p_m^2."""
    p, sizes = basins(F)
    n = F.size
    return float(np.sum((sizes/n)**2))

def gaussian_field(L, S, rho, rng):
    """Real Gaussian field on (S,)*L with covariance rho^{d(x,y)}.
    Per-mode variance a^k b^(L-k), a=(1-rho)/S, b=rho+(1-rho)/S.
    Spectral synthesis enforcing phi_{-k}=conj(phi_k)."""
    a=(1.0-rho)/S; b=rho+(1.0-rho)/S
    V=np.array([(a**k)*(b**(L-k)) for k in range(L+1)],float)
    keys=list(product(range(S),repeat=L))
    P=np.zeros((S,)*L,dtype=complex)
    seen=set()
    for k in keys:
        if k in seen: continue
        neg=tuple((-x)%S for x in k)
        o=sum(1 for x in k if x!=0)
        if neg==k:
            P[k]=rng.normal(0,np.sqrt(V[o])); seen.add(k)
        else:
            z=rng.normal(0,np.sqrt(V[o]/2))+1j*rng.normal(0,np.sqrt(V[o]/2))
            P[k]=z; P[neg]=np.conj(z); seen.add(k); seen.add(neg)
    return iwht(P)

def empirical_rho1(F, n_pairs=200000, rng=None):
    """Correlation between fitness of random Hamming-distance-1 pairs."""
    shape = F.shape; L = len(shape); S = shape[0]
    rng = rng or np.random.default_rng(0)
    n = F.size
    a = rng.integers(0, n, n_pairs)
    coords = np.array(np.unravel_index(a, shape)).T
    site = rng.integers(0, L, n_pairs)
    alt = rng.integers(0, S, n_pairs)
    # ensure alt != current
    cur = coords[np.arange(n_pairs), site]
    alt = np.where(alt == cur, (alt+1) % S, alt)
    coords2 = coords.copy(); coords2[np.arange(n_pairs), site] = alt
    b = np.ravel_multi_index(tuple(coords2.T), shape)
    x = F.ravel()[a]; y = F.ravel()[b]
    return float(np.corrcoef(x, y)[0, 1])

def reachable_bfs(F, sources=None, max_iter=100000, verbose=False):
    """Correct directed reachability: BFS from sources down the uphill DAG."""
    shape=F.shape; L=len(shape); S=shape[0]
    R=np.zeros(shape,dtype=bool)
    if sources is None:
        R.flat[int(np.argmax(F))]=True
    else:
        R.flat[np.asarray(sources,dtype=int)]=True
    frontier=R.copy()
    it=0
    while frontier.any() and it<max_iter:
        new=np.zeros(shape,dtype=bool)
        for a in range(L):
            for b in range(S):
                sl=[slice(None)]*L; sl[a]=b
                Fb=F[tuple(sl)]; Rb=frontier[tuple(sl)]
                if not Rb.any(): continue
                for b2 in range(S):
                    if b2==b: continue
                    sl2=[slice(None)]*L; sl2[a]=b2
                    lower=F[tuple(sl2)]<Fb
                    new[tuple(sl2)] |= Rb & lower
        new &= ~R
        R |= new; frontier=new; it+=1
    if verbose: print("bfs iters",it)
    return R

def reachable_consistent(F, Fp, sources=None):
    """Reachable set of the global max in D_F, restricted to edges whose
    orientation is the SAME in F and Fp.  Equals A(D_F \ F_flip)."""
    shape=F.shape; L=len(shape); S=shape[0]
    R=np.zeros(shape,dtype=bool)
    if sources is None: R.flat[int(np.argmax(F))]=True
    else: R.flat[np.asarray(sources,dtype=int)]=True
    frontier=R.copy()
    while frontier.any():
        new=np.zeros(shape,dtype=bool)
        for a in range(L):
            for b in range(S):
                sl=[slice(None)]*L; sl[a]=b
                Rb=frontier[tuple(sl)]
                if not Rb.any(): continue
                for b2 in range(S):
                    if b2==b: continue
                    sl2=[slice(None)]*L; sl2[a]=b2
                    lower_F = F[tuple(sl2)] < F[tuple(sl)]
                    same    = (Fp[tuple(sl2)] < Fp[tuple(sl)]) == lower_F
                    new[tuple(sl2)] |= Rb & lower_F & same
        new &= ~R
        R |= new; frontier=new
    return R

def flipped_edge_count(F, Fp):
    """Number of undirected edges whose orientation differs between F and Fp."""
    shape=F.shape; L=len(shape); S=shape[0]; tot=0
    for a in range(L):
        for b in range(S):
            sl=[slice(None)]*L; sl[a]=b
            for b2 in range(b+1,S):
                sl2=[slice(None)]*L; sl2[a]=b2
                d1=F[tuple(sl2)]-F[tuple(sl)]
                d2=Fp[tuple(sl2)]-Fp[tuple(sl)]
                tot += int(np.sum((d1>0)!=(d2>0)))
    return tot
