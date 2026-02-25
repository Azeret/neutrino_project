# neutrino_project (toy model): carbon-burning red supergiants + IMF sensitivity

This repository is designed to be **simple and transparent**:
- it estimates the expected number of **carbon-burning red supergiants (C-burning RSGs)** in the Milky Way
- it compares how that number changes with different **IMFs**

It is a **toy population model**, not a full reproduction of `neutrinos.pdf` (Seong et al. 2025), which uses MESA neutrino emissivities and detector simulations.

## The fastest way to run (recommended)

Open `run_pipeline.py`, adjust the **CONFIG** block if you want, then run:

```bash
python3 run_pipeline.py
```

Outputs are written to `outputs/`:
- `outputs/imf_scan.csv`: table of expected counts per IMF (main column: `cburn_rsg_mean`)
- `outputs/imf_scan.png`: bar plot of expected **C-burning RSG** counts per IMF
- `outputs/mw_snapshot_map.png`: 2D toy Milky Way snapshot (phase-colored)
- `outputs/counts_within_radius_vs_time.png`: expected RSG / C-burning RSG counts within 1 kpc vs time
- `outputs/toy_neutrino_yield_vs_time.png`: toy flux + toy event-rate vs time
- `outputs/phase_timeline_18msun.png`: how phase windows are estimated for one mass
- `outputs/isochrone_hrd_rsg_cut.png`: CMD 3.9 isochrone HR diagram + RSG cut (pedagogical)

If you get `ModuleNotFoundError` (e.g. missing `numpy`), install dependencies:

```bash
pip install -r requirements.txt
```

## Example plots (already generated)

The folder `figures/` contains example outputs for the default `run_pipeline.py` CONFIG,
so you can see what the plots look like without running anything.

### 2D Milky Way snapshot (toy)

![](figures/mw_snapshot_map.png)

### Counts within 1 kpc vs time (toy)

![](figures/counts_within_radius_vs_time.png)

### Toy neutrino flux + events vs time

![](figures/toy_neutrino_yield_vs_time.png)

### Phase-window illustration (one mass)

![](figures/phase_timeline_18msun.png)

### Isochrone HR diagram + RSG cut (pedagogical)

![](figures/isochrone_hrd_rsg_cut.png)

## What “C-burning RSG” means in this project

We define:

- **RSG (toy definition)**: `Teff <= 4000 K` and `log10(L/Lsun) >= 4.5`
- **C-burning (toy definition)**: PARSEC track column `LC >= 1e-3`
- **C-burning RSG**: overlap of the above two conditions

These definitions are intentionally simple and are meant for sensitivity studies.

## Data files

The repo includes small precomputed phase-window tables:
- `data/parsec/v2_vms/phases_Z0p014.csv`
- `data/parsec/v2_vms/phases_Z0p017.csv`

These contain (per mass):
- lifetime
- C-burning window duration
- RSG duration
- overlap duration (C-burning AND RSG)

### What to change in `run_pipeline.py`

Common “knobs” in the CONFIG block:
- `phases_csv`: switch between `phases_Z0p014.csv` and `phases_Z0p017.csv`
- `imfs`: choose which IMFs to compare
- `sfr_msun_per_yr`, `duration_myr`: star formation assumptions
- `n_sims`: Monte Carlo precision (bigger = smoother, slower)
- `make_*`: turn individual plots on/off

The **large PARSEC track ZIPs** are **not** included (they are ~85–95 MB each). If you obtain them yourself, you can set:
`recompute_phases_from_zip=True` in `run_pipeline.py` to regenerate the phase CSV.

This repo also includes small CMD 3.9 isochrone tables (for plotting):
- `data/parsec/isochrones/parsec_cmd39_v1p2s_Z0p0152_logAge7p0.dat`
- `data/parsec/isochrones/parsec_cmd39_v2p0_Z0p0152_logAge7p0.dat`

## Optional: advanced CLI (uses flags)

If you later want a more flexible interface:

```bash
python3 -m neutrino_project --help
```

Useful commands:
- `extract-phases`: build a phases CSV from a PARSEC track ZIP
- `imf-scan`: compare IMFs and get the expected C-burning RSG counts
- `plot-hr`: plot an HR diagram from one track and highlight RSG/C-burning regions

## How this relates to neutrinos.pdf (Seong et al. 2025)

This project focuses on **population counts** (how many C-burning RSGs exist, and how that depends on the IMF).

To move toward reproducing the paper’s detectability calculations:
- keep the population backbone here
- replace the toy neutrino part with a calibrated `L_ν(t, M, Z)` and spectra from MESA (or the paper’s tables/figures)

Implementation notes are in `neutrino_project/neutrinos.py`.
