#!/bin/sh
# Reproduce every result and figure in the paper (after ./fetch_data.sh). About 15 minutes on a laptop.
set -e
cd "$(dirname "$0")"
python3 robust_gb1.py          # GB1: models, thresholds, bootstrap, Poisson nulls  -> robust_results.json
python3 robust_depth.py        # GB1: depth simulation and thinning                 -> robust_depth_results.json
python3 robust_folA.py         # folA: edges, data, nulls, thinning                 -> robust_folA_results.json
python3 robust_folA_grid.py    # folA: noise-scale grid                             -> robust_folA_grid_results.json
python3 robust_folA_degrade.py # folA: single-replicate / GB1-depth degradation     -> robust_folA_degrade_results.json
python3 joint_tests.py         # joint (Mahalanobis) tests vs nulls                -> joint_tests_results.json
python3 fig_noise.py           # Figure 1
python3 fig_folA.py            # Figure 2
if [ -f paper.tex ]; then pdflatex -interaction=nonstopmode paper.tex >/dev/null && pdflatex -interaction=nonstopmode paper.tex >/dev/null; fi
