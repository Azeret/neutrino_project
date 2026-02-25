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
from neutrino_project.plots import (
    plot_parsec_hrd,
    plot_cmd39_isochrone_hrd,
    plot_counts_within_radius_vs_time,
    plot_mw_snapshot_map,
    plot_neutrino_yield_vs_time,
    plot_phase_timeline,
)
from neutrino_project.detectability import detector_presets, estimate_detectability_at_time
from neutrino_project.imf_constraint import default_future_detector, imf_constraint_matrix
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

    # Plots (set to False to skip)
    make_mw_map: bool = True
    make_time_counts: bool = True
    make_time_neutrinos: bool = True
    make_phase_timeline: bool = True
    make_isochrone_plot: bool = True
    make_track_hrd_example: bool = True
    make_detectability_table: bool = True
    make_detectability_plot: bool = True
    make_imf_constraint_demo: bool = True

    # Neutrino toy parameters used in the time-evolution plot
    # (order-of-magnitude only; for a paper-level model see neutrino_project/neutrinos.py)
    lnu_per_star_erg_s: float = 1e38
    mean_energy_mev: float = 0.9
    alpha: float = 2.0
    detector_kton: float = 22.5
    inv_d2_samples: int = 100_000
    reference_distance_pc: float = 200.0
    constraint_exposure_years: float = 10.0
    constraint_z_sigma: float = 3.0
    constraint_imfs: tuple[str, ...] = ("kroupa", "top-light", "top-heavy")
    constraint_method: str = "asimov"  # "asimov" (LLR) or "gaussian"
    constraint_include_background: bool = True
    # Background model: background scales linearly with detector mass (kton).
    # This is an "effective" residual background after cuts, in events per (kton*year).
    # Set to 0.0 (or constraint_include_background=False) to remove backgrounds entirely.
    constraint_background_per_kton_year: float = 1.0

    # Optional: isochrone file for the HR plot (included in this repo)
    isochrone_dat: Path = Path("data/parsec/isochrones/parsec_cmd39_v1p2s_Z0p0152_logAge7p0.dat")

    # Optional: track HRD example (only works if you downloaded the PARSEC ZIP).
    track_hrd_mass_msun: float = 18.0


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

    if CFG.make_detectability_table:
        out_det_csv = CFG.out_dir / "detectability_imf.csv"
        dets = detector_presets()
        # Override the SK-like mass from CONFIG while keeping the rest of the preset list.
        dets = [
            d.__class__(name=f"SK-like ({CFG.detector_kton:g} kt)", fiducial_mass_kton=CFG.detector_kton)
            if d.name.startswith("SK-like")
            else d
            for d in dets
        ]

        import csv

        rows = []
        for imf in CFG.imfs:
            r = estimate_detectability_at_time(
                phases_csv=CFG.phases_csv,
                imf=imf,
                sfr_msun_per_yr=CFG.sfr_msun_per_yr,
                t_obs_myr=CFG.duration_myr,
                radius_kpc=CFG.radius_kpc,
                sun_xy_kpc=(CFG.sun_x_kpc, CFG.sun_y_kpc),
                within_samples=CFG.within_samples,
                inv_d2_samples=CFG.inv_d2_samples,
                seed=CFG.seed,
                lnu_per_star_erg_s=CFG.lnu_per_star_erg_s,
                mean_energy_mev=CFG.mean_energy_mev,
                alpha=CFG.alpha,
                detector_list=dets,
                reference_distance_pc=CFG.reference_distance_pc,
            )
            rows.append(r)

        out_det_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_det_csv.open("w", newline="") as f:
            fieldnames = [
                "imf",
                "t_myr",
                "radius_kpc",
                "p_within",
                "n_cburn_rsg_mw",
                "n_cburn_rsg_within",
                "flux_within_cm2_s",
                "reference_distance_pc",
                "flux_one_star_at_refdist_cm2_s",
                *[f"events_per_year__{d.name}" for d in dets],
                *[f"events_per_year__one_star_at_refdist__{d.name}" for d in dets],
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for imf, r in zip(CFG.imfs, rows, strict=True):
                row = {
                    "imf": imf,
                    "t_myr": f"{r.t_myr:.6g}",
                    "radius_kpc": f"{CFG.radius_kpc:.6g}",
                    "p_within": f"{r.p_within:.6g}",
                    "n_cburn_rsg_mw": f"{r.n_cburn_rsg_mw:.6g}",
                    "n_cburn_rsg_within": f"{r.n_cburn_rsg_within:.6g}",
                    "flux_within_cm2_s": f"{r.flux_within_cm2_s:.6g}",
                    "reference_distance_pc": f"{CFG.reference_distance_pc:.6g}",
                    "flux_one_star_at_refdist_cm2_s": f"{r.flux_one_star_at_refdist_cm2_s:.6g}",
                }
                for d in dets:
                    row[f"events_per_year__{d.name}"] = f"{r.events_per_year_by_detector[d.name]:.6g}"
                    row[f"events_per_year__one_star_at_refdist__{d.name}"] = f"{r.events_per_year_one_star_at_refdist[d.name]:.6g}"
                w.writerow(row)

        print(f"Saved: {out_det_csv}")

        if CFG.make_detectability_plot:
            import os

            tmp = Path(os.environ.get("TMPDIR", "/tmp"))
            Path(os.environ.setdefault("MPLCONFIGDIR", str(tmp / "matplotlib"))).mkdir(parents=True, exist_ok=True)
            Path(os.environ.setdefault("XDG_CACHE_HOME", str(tmp / "xdg_cache"))).mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt
            import numpy as np

            out_det_png = CFG.out_dir / "detectability_imf_events.png"
            xs = np.arange(len(CFG.imfs))
            width = min(0.16, 0.8 / max(1, len(dets)))
            fig, ax = plt.subplots(figsize=(9.6, 4.8))

            for j, d in enumerate(dets):
                y = [r.events_per_year_by_detector[d.name] for r in rows]
                ax.bar(xs + (j - (len(dets) - 1) / 2) * width, y, width=width, label=d.name)

            ax.set_xticks(xs)
            ax.set_xticklabels(list(CFG.imfs), rotation=0)
            ax.set_ylabel("Toy ES events/year (signal-only)")
            ax.set_yscale("log")
            ax.axhline(1.0, color="0.5", lw=1.0, ls="--")
            ax.text(xs[0] - 0.4, 1.15, "≈1 event/yr", color="0.4", fontsize=8)
            ax.set_title(f"Detectability vs IMF (within {CFG.radius_kpc:g} kpc)")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(out_det_png, dpi=200)
            plt.close(fig)

            print(f"Saved: {out_det_png}")

    if CFG.make_imf_constraint_demo:
        # Question: if detection were feasible, could the rate constrain the IMF?
        # Here we compute a simple required detector mass (kton water-equivalent)
        # to separate IMF pairs at z_sigma after an exposure.
        #
        # Important: this treats the SFR and the phase-window model as fixed inputs.
        # In a more complete analysis they would be inferred jointly (with EM priors, e.g. Gaia),
        # which can reduce or increase IMF discriminability depending on degeneracies.
        out_matrix_csv = CFG.out_dir / "imf_constraint_required_mass.csv"
        out_matrix_png = CFG.out_dir / "imf_constraint_required_mass.png"

        det = default_future_detector()
        res_list, mat = imf_constraint_matrix(
            phases_csv=CFG.phases_csv,
            imfs=list(CFG.constraint_imfs),
            detector=det,
            exposure_years=CFG.constraint_exposure_years,
            z_sigma=CFG.constraint_z_sigma,
            include_background=CFG.constraint_include_background,
            background_per_kton_year=CFG.constraint_background_per_kton_year,
            method=CFG.constraint_method,
            sfr_msun_per_yr=CFG.sfr_msun_per_yr,
            t_obs_myr=CFG.duration_myr,
            radius_kpc=CFG.radius_kpc,
            sun_xy_kpc=(CFG.sun_x_kpc, CFG.sun_y_kpc),
            within_samples=CFG.within_samples,
            inv_d2_samples=CFG.inv_d2_samples,
            seed=CFG.seed,
            lnu_per_star_erg_s=CFG.lnu_per_star_erg_s,
            mean_energy_mev=CFG.mean_energy_mev,
            alpha=CFG.alpha,
        )

        import csv

        with out_matrix_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "imf_a",
                    "imf_b",
                    "method",
                    "detector_name",
                    "exposure_years",
                    "z_sigma",
                    "include_background",
                    "background_per_kton_year",
                    "base_background_per_year",
                    "base_mass_kton",
                    "base_rate_a_per_year",
                    "base_rate_b_per_year",
                    "required_mass_kton_for_z",
                ],
            )
            w.writeheader()
            for r in res_list:
                w.writerow(r.__dict__)

        # Plot a heatmap of required mass (kton) to separate IMF pairs.
        import os

        tmp = Path(os.environ.get("TMPDIR", "/tmp"))
        Path(os.environ.setdefault("MPLCONFIGDIR", str(tmp / "matplotlib"))).mkdir(parents=True, exist_ok=True)
        Path(os.environ.setdefault("XDG_CACHE_HOME", str(tmp / "xdg_cache"))).mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        # Use log10 scale for readability; cap huge values.
        mat_plot = np.log10(np.clip(mat, 1e-3, 1e13))
        im = ax.imshow(mat_plot, cmap="viridis")
        ax.set_xticks(range(len(CFG.constraint_imfs)))
        ax.set_yticks(range(len(CFG.constraint_imfs)))
        ax.set_xticklabels(list(CFG.constraint_imfs), rotation=30, ha="right")
        ax.set_yticklabels(list(CFG.constraint_imfs))
        ax.set_title(
            f"Required detector mass to separate IMFs (toy, ES-only)\n"
            f"{det.name}, T={CFG.constraint_exposure_years:g} yr, z={CFG.constraint_z_sigma:g}, "
            f"method={CFG.constraint_method}"
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("log10(required mass [kton])")
        fig.tight_layout()
        fig.savefig(out_matrix_png, dpi=200)
        plt.close(fig)

        print(f"Saved: {out_matrix_csv}")
        print(f"Saved: {out_matrix_png}")

    # Extra plots (all optional, controlled by CONFIG flags)
    if CFG.make_mw_map:
        out = CFG.out_dir / "mw_snapshot_map.png"
        plot_mw_snapshot_map(
            phases_csv=CFG.phases_csv,
            out_png=out,
            imf=CFG.imfs[0],
            sfr_msun_per_yr=CFG.sfr_msun_per_yr,
            t_obs_myr=CFG.duration_myr,
            seed=CFG.seed,
            radius_kpc=CFG.radius_kpc,
            sun_xy_kpc=(CFG.sun_x_kpc, CFG.sun_y_kpc),
        )
        print(f"Saved: {out}")

    if CFG.make_time_counts:
        out = CFG.out_dir / "counts_within_radius_vs_time.png"
        plot_counts_within_radius_vs_time(
            phases_csv=CFG.phases_csv,
            out_png=out,
            imf=CFG.imfs[0],
            sfr_msun_per_yr=CFG.sfr_msun_per_yr,
            t_max_myr=max(CFG.duration_myr, 25.0),
            radius_kpc=CFG.radius_kpc,
            sun_xy_kpc=(CFG.sun_x_kpc, CFG.sun_y_kpc),
            within_samples=CFG.within_samples,
            seed=CFG.seed,
        )
        print(f"Saved: {out}")

    if CFG.make_time_neutrinos:
        out = CFG.out_dir / "toy_neutrino_yield_vs_time.png"
        plot_neutrino_yield_vs_time(
            phases_csv=CFG.phases_csv,
            out_png=out,
            imf=CFG.imfs[0],
            sfr_msun_per_yr=CFG.sfr_msun_per_yr,
            t_max_myr=max(CFG.duration_myr, 25.0),
            radius_kpc=CFG.radius_kpc,
            sun_xy_kpc=(CFG.sun_x_kpc, CFG.sun_y_kpc),
            within_samples=CFG.within_samples,
            seed=CFG.seed,
            lnu_per_star_erg_s=CFG.lnu_per_star_erg_s,
            mean_energy_mev=CFG.mean_energy_mev,
            alpha=CFG.alpha,
            detector_kton=CFG.detector_kton,
        )
        print(f"Saved: {out}")

    if CFG.make_phase_timeline:
        out = CFG.out_dir / "phase_timeline_18msun.png"
        plot_phase_timeline(
            phases_csv=CFG.phases_csv,
            mass_msun=18.0,
            out_png=out,
        )
        print(f"Saved: {out}")

    if CFG.make_isochrone_plot and CFG.isochrone_dat.exists():
        out = CFG.out_dir / "isochrone_hrd_rsg_cut.png"
        plot_cmd39_isochrone_hrd(
            isochrone_dat=CFG.isochrone_dat,
            out_png=out,
        )
        print(f"Saved: {out}")

    if CFG.make_track_hrd_example and CFG.parsec_zip.exists():
        out = CFG.out_dir / f"track_hrd_{CFG.track_hrd_mass_msun:g}msun.png"
        plot_parsec_hrd(
            zip_path=CFG.parsec_zip,
            mass_msun=CFG.track_hrd_mass_msun,
            out_png=out,
            lc_threshold=CFG.lc_threshold,
            rsg_teff_max_k=CFG.rsg_teff_max_k,
            rsg_logl_min=CFG.rsg_logl_min,
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
