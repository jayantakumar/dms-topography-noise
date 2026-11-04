"""GB1 read counts as tensors, Poisson fitness errors, and the global optimum of the data."""
import numpy as np, pandas as pd, robust

ALPH = "ACDEFGHIKLMNPQRSTVWY"; S = 20; L = 4; IDX = {a: i for i, a in enumerate(ALPH)}
df = pd.read_csv('data/four_mutations_full_data.csv',
                 usecols=['Variants', 'Count input', 'Count selected'])
sel = np.zeros((S,)*L); inp = np.ones((S,)*L); mask = np.zeros((S,)*L, bool)
for v, ci, cs in zip(df.Variants, df['Count input'], df['Count selected']):
    x = tuple(IDX[c] for c in v); inp[x] = ci; sel[x] = cs; mask[x] = True
wt = tuple(IDX[c] for c in "VDGV"); norm = inp[wt]/sel[wt]

def fitness(sel, inp): return norm*sel/inp
def stderr(sel, inp): return norm*np.sqrt(sel + 1)/inp      # Poisson, floored at one read

F = fitness(sel, inp); SE = stderr(sel, inp)
target = int(np.argmax(np.where(mask, F, -np.inf)))
order = robust.order_index(L, S)
