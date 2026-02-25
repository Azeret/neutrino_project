"""
One-file runner (no command-line flags).

Goal
----
Main output: a small table (and plot) of the expected number of **carbon-burning
red supergiants (C-burning RSGs)** in the Milky Way for different IMFs.

How to run
----------
1) (Optional) create a clean environment and install requirements:
     pip install -r requirements.txt
2) Run:
     python3 run_pipeline.py

What to edit
------------
Edit the CONFIG section below (paths, SFR, IMFs, etc.).

Notes
-----
This is a toy population model:
- phase durations come from PARSEC tracks (or the precomputed CSV in data/)
- neutrino event-rate utilities are intentionally simplified

For the detailed MESA-based neutrino model used in neutrinos.pdf (Seong et al. 2025),
see the notes in neutrino_project/neutrinos.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neutrino_project.parsec_v2_vms import extract_phase_windows_to_csv
from neutrino_project.population import run_imf_scan


# =========================
# CONFIG (edit this section)
# =========================


@dataclass(frozen=True)
class Config:
    # If you do NOT have the PARSEC track ZIPs, keep recompute_phases=False and
    # use the precomputed phases CSVs shipped in data/parsec/v2_vms/.
    recompute_phases_from_zip: bool = False

    # PARSEC track ZIP (not included in the repo; see README for notes).
    parsec_zip: Path = Path("data/parsec/v2_vms/Z0.014_Y0.273_tracks.zip")

    # Masses to include (must exist in the ZIP).
    masses_msun: tuple[float, ...] = (12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35)

    # Phase definitions (toy, but transparent)
    lc_threshold: float = 1e-3
    rsg_teff_max_k: float = 4000.0
    rsg_logl_min: float = 4.5

    # Use either:
    # - a precomputed phases CSV (default)
    # - or recompute from parsec_zip into this same path
    phases_csv: Path = Path("data/parsec/v2_vms/phases_Z0p014.csv")

    # MW toy population assumptions
    imfs: tuple[str, ...] = ("kroupa", "salpeter", "top-heavy", "top-light")
    sfr_msun_per_yr: float = 2.0
    duration_myr: float = 20.0

    # Monte Carlo settings (increase n_sims for smoother results)
    n_sims: int = 10_000
    seed: int = 1

    # Optional "near the Sun" calculation
    radius_kpc: float = 1.0
    sun_x_kpc: float = -8.2
    sun_y_kpc: float = 0.0
    within_samples: int = 200_000

    fast: bool = True

    # Outputs
    out_dir: Path = Path("outputs")


CFG = Config()


def main() -> None:
    CFG.out_dir.mkdir(parents=True, exist_ok=True)

    if CFG.recompute_phases_from_zip:
        if not CFG.parsec_zip.exists():
            raise SystemExit(
                "recompute_phases_from_zip=True but the PARSEC ZIP was not found:\n"
                f"  {CFG.parsec_zip}\n"
                "Either download the ZIP, or set recompute_phases_from_zip=False."
            )
        print(f"[1/2] Extracting phase windows from {CFG.parsec_zip}")
        extract_phase_windows_to_csv(
            zip_path=CFG.parsec_zip,
            masses_msun=[float(m) for m in CFG.masses_msun],
            out_csv=CFG.phases_csv,
            lc_threshold=CFG.lc_threshold,
            rsg_teff_max_k=CFG.rsg_teff_max_k,
            rsg_logl_min=CFG.rsg_logl_min,
        )
        print(f"Saved phases CSV: {CFG.phases_csv}")
    else:
        if not CFG.phases_csv.exists():
            raise SystemExit(
                "Phases CSV not found:\n"
                f"  {CFG.phases_csv}\n"
                "Fix: either point CFG.phases_csv to an existing file, or set "
                "recompute_phases_from_zip=True and provide CFG.parsec_zip."
            )

    print("[2/2] Running IMF scan (main result: expected C-burning RSG count)")
    out_csv = CFG.out_dir / "imf_scan.csv"
    out_plot = CFG.out_dir / "imf_scan.png"
    run_imf_scan(
        phases_csv=CFG.phases_csv,
        imfs=list(CFG.imfs),
        sfr_msun_per_yr=CFG.sfr_msun_per_yr,
        duration_myr=CFG.duration_myr,
        n_sims=CFG.n_sims,
        seed=CFG.seed,
        radius_kpc=CFG.radius_kpc,
        sun_x_kpc=CFG.sun_x_kpc,
        sun_y_kpc=CFG.sun_y_kpc,
        within_samples=CFG.within_samples,
        fast=CFG.fast,
        out_csv=out_csv,
        out_plot=out_plot,
    )
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_plot}")


if __name__ == "__main__":
    main()

