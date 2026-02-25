# Precomputed phase windows (small CSVs)

The CSV files in this folder are meant to let you run `run_pipeline.py` without
downloading the full PARSEC track ZIPs.

Files:
- `phases_Z0p014.csv`: extracted from PARSEC v2.0 VMS grid at Z≈0.014
- `phases_Z0p017.csv`: extracted from PARSEC v2.0 VMS grid at Z≈0.017

Each row corresponds to one initial mass and contains:
- `lifetime_myr`: track age of the last tabulated model
- `cburn_duration_kyr`: total time with `LC >= lc_threshold`
- `rsg_duration_myr`: total time satisfying the toy RSG cut (`Teff` + `logL`)
- `cburn_rsg_duration_kyr`: overlap of C-burning and RSG

If you obtain the original PARSEC track ZIPs, you can regenerate these tables
using either:
- `run_pipeline.py` with `recompute_phases_from_zip=True`, or
- `python3 -m neutrino_project extract-phases ...`

