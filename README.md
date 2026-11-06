# Measurement noise can mimic higher-order epistasis in the topography of fitness landscapes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24008996.svg)](https://doi.org/10.5281/zenodo.24008996)

Analysis code for the manuscript of the same title (in preparation).

Peaks, basins and evolutionary accessibility of an empirical fitness landscape depend only on which of
each pair of neighbouring genotypes is fitter. Most of those comparisons are not resolved by deep
mutational scanning data. This code measures how much of the reported topography of two combinatorially
complete landscapes survives measurement noise:

- **GB1** (Wu et al. 2016, 20^4 genotypes, one selection): peak and basin statistics are not converged at the
  experimental depth and are reproduced by low-order models plus counting noise.
- **folA** (Papkou et al. 2023, 4^9 genotypes, six replicate selections): statistics converge, and the basin
  concentration cannot be reproduced by low-order models plus noise, even after degrading the data to a
  single replicate at GB1's read depth.

## Reproduce

```sh
pip install -r requirements.txt   # numpy, pandas, matplotlib (versions used: see file)
./fetch_data.sh                   # downloads both datasets into data/ and verifies SHA-256 checksums
./run_all.sh                      # all results and both figures (~15 min)
```

## Files

| file | what it does |
|---|---|
| `common.py` | `nb_slice`: neighbour alignment on the Hamming-graph tensor |
| `robust.py` | significance-thresholded reachability, steepest-ascent basins, peaks, order-k Fourier fits on observed genotypes |
| `robust_gb1_data.py`, `robust_gb1.py` | GB1 counts, Poisson errors; models, thresholds, bootstrap, null models |
| `robust_depth.py` | GB1 depth simulation and binomial thinning |
| `robust_folA_data.py`, `robust_folA.py` | folA fitness, replicate errors, counts; edges, nulls, thinning |
| `robust_folA_grid.py` | folA noise-scale grid for the null models |
| `robust_folA_degrade.py` | folA degraded to one replicate and to GB1 read depth |
| `fig_noise.py`, `fig_folA.py` | Figures 1 and 2 |
| `robust*_results.json` | results reported in the paper |

All scripts use fixed random seeds; `run_all.sh` regenerates the committed result files byte for byte.

## Data

Not redistributed here; `fetch_data.sh` downloads them from their sources.

- GB1: Wu NC et al. (2016) *eLife* 5:e16965, doi:10.7554/eLife.16965, via the FLIP benchmark
  (github.com/J-SNACKKB/FLIP).
- folA: Papkou A et al. (2023) *Science* 382:eadh3860, doi:10.1126/science.adh3860; data from the authors'
  Zenodo deposit, doi:10.5281/zenodo.13837982 (CC-BY-4.0).

## Cite

Kumar J. *Measurement noise can mimic higher-order epistasis in the topography of fitness landscapes: analysis code.*
Zenodo (2026). doi:[10.5281/zenodo.24008996](https://doi.org/10.5281/zenodo.24008996)
(this DOI always resolves to the latest version; v1.0.1 is doi:10.5281/zenodo.24008997).

## License

Code: MIT (see `LICENSE`).
