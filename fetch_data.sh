#!/bin/sh
# Download the two public datasets into data/ and verify them against the exact files used in the paper.
#   GB1  (Wu et al. 2016) via the FLIP benchmark: github.com/J-SNACKKB/FLIP
#   folA (Papkou et al. 2023) from the authors' Zenodo deposit, doi:10.5281/zenodo.13837982 (CC-BY-4.0)
set -e
cd "$(dirname "$0")"
mkdir -p data/folA_raw
curl -sL -o data/gb1.zip https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/four_mutations_full_data.csv.zip
unzip -o -q data/gb1.zip four_mutations_full_data.csv -d data && rm data/gb1.zip
for f in fitness_data_wt.csv sequence_reads_counts_after_selection.csv sequence_reads_counts_before_selection.csv; do
    curl -sL -o data/folA_raw/$f https://zenodo.org/api/records/13837982/files/$f/content
done
shasum -a 256 -c <<'SUMS'
14cfb23e04edcaf99daa106743782b9289c1c673fcb6193ed90f8f19d33755f3  data/four_mutations_full_data.csv
471ad72e2d735b3bbd87233201674480fa01cb35fe63082242dcba4f7ee334d1  data/folA_raw/fitness_data_wt.csv
37703510f242f1680672a97cb0ed70aec1d369fc21b48efecd7519d2a27b6635  data/folA_raw/sequence_reads_counts_after_selection.csv
1aaaa16662e4fb0cf3d3e3e98fb054b6dc417c4c18e26df57a713f42b44a68a6  data/folA_raw/sequence_reads_counts_before_selection.csv
SUMS
