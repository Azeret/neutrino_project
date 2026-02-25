from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .neutrinos import Detector, NeutrinoSpectrumModel, estimate_es_events_per_year
from .parsec_v2_vms import extract_phase_windows_to_csv, find_track_members
from .plots import plot_alpha_spectrum, plot_parsec_hrd
from .population import PopulationInputs, run_imf_scan, run_population_mc


def _cmd_extract(args: argparse.Namespace) -> int:
    extract_phase_windows_to_csv(
        zip_path=args.zip,
        masses_msun=args.mass,
        out_csv=args.out,
        lc_threshold=args.lc_threshold,
        rsg_teff_max_k=args.rsg_teff_max_k,
        rsg_logl_min=args.rsg_logl_min,
    )
    return 0


def _cmd_population(args: argparse.Namespace) -> int:
    inputs = PopulationInputs(
        phases_csv=args.phases_csv,
        imf=args.imf,
        sfr_msun_per_yr=args.sfr,
        duration_myr=args.duration_myr,
        n_sims=args.n_sims,
        seed=args.seed,
        radius_kpc=args.radius_kpc,
        sun_x_kpc=args.sun_x,
        sun_y_kpc=args.sun_y,
        within_samples=args.within_samples,
        fast=args.fast,
    )
    run_population_mc(inputs, out_json=args.out_json, out_plot=args.out_plot)
    return 0


def _cmd_imf_scan(args: argparse.Namespace) -> int:
    imfs = args.imf or ["kroupa", "salpeter", "top-heavy", "top-light"]
    run_imf_scan(
        phases_csv=args.phases_csv,
        imfs=imfs,
        sfr_msun_per_yr=args.sfr,
        duration_myr=args.duration_myr,
        n_sims=args.n_sims,
        seed=args.seed,
        radius_kpc=args.radius_kpc,
        sun_x_kpc=args.sun_x,
        sun_y_kpc=args.sun_y,
        within_samples=args.within_samples,
        fast=args.fast,
        out_csv=args.out_csv,
        out_plot=args.out_plot,
    )
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    det = Detector.water_equivalent(
        fiducial_mass_kton=args.detector_kton,
        name=args.detector_name,
    )
    spec = NeutrinoSpectrumModel.alpha_fit(
        mean_energy_mev=args.mean_energy_mev,
        alpha=args.alpha,
    )
    events = estimate_es_events_per_year(
        flux_at_earth_cm2_s=args.flux_cm2_s,
        detector=det,
        spectrum=spec,
        e_min_mev=args.e_min_mev,
        e_max_mev=args.e_max_mev,
        n_grid=args.n_grid,
        flavor=args.flavor,
    )
    print(
        f"{det.name}: {events:.3g} ES events/year "
        f"(flux={args.flux_cm2_s:.3g} cm^-2 s^-1, <E>={args.mean_energy_mev:g} MeV)"
    )
    return 0


def _cmd_list_masses(args: argparse.Namespace) -> int:
    members = find_track_members(args.zip)
    ms = sorted(members)
    if not ms:
        raise SystemExit("No track members found (unexpected filename format?)")
    print(f"{len(ms)} masses in {args.zip}:")
    print(" ".join(f"{m:g}" for m in ms))
    return 0


def _cmd_plot_hr(args: argparse.Namespace) -> int:
    plot_parsec_hrd(
        zip_path=args.zip,
        mass_msun=args.mass,
        out_png=args.out,
        lc_threshold=args.lc_threshold,
        rsg_teff_max_k=args.rsg_teff_max_k,
        rsg_logl_min=args.rsg_logl_min,
    )
    return 0


def _cmd_plot_spectrum(args: argparse.Namespace) -> int:
    plot_alpha_spectrum(
        mean_energy_mev=args.mean_energy_mev,
        alpha=args.alpha,
        out_png=args.out,
        e_min_mev=args.e_min_mev,
        e_max_mev=args.e_max_mev,
        n_grid=args.n_grid,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m neutrino_project",
        description="Toy MW population model for carbon-burning red supergiants (IMF sensitivity).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser(
        "extract-phases",
        help="Extract lifetimes, C-burning, RSG windows from PARSEC v2.0 VMS track ZIP to CSV.",
    )
    ex.add_argument("--zip", type=Path, required=True, help="Path to PARSEC v2.0 VMS tracks .zip")
    ex.add_argument("--mass", type=float, action="append", required=True, help="Initial mass (repeatable)")
    ex.add_argument("--out", type=Path, required=True, help="Output CSV path")
    ex.add_argument("--lc-threshold", type=float, default=1e-3, help="Define C-burning as LC >= threshold")
    ex.add_argument("--rsg-teff-max-k", type=float, default=4000.0, help="RSG criterion: Teff <= this (K)")
    ex.add_argument(
        "--rsg-logl-min",
        type=float,
        default=4.5,
        help="RSG criterion: log10(L/Lsun) >= this",
    )
    ex.set_defaults(func=_cmd_extract)

    pop = sub.add_parser("population", help="Monte Carlo MW population counts from a phases CSV + IMF.")
    pop.add_argument("--phases-csv", type=Path, required=True)
    pop.add_argument("--imf", choices=["kroupa", "salpeter", "top-heavy", "top-light"], default="kroupa")
    pop.add_argument("--sfr", type=float, default=2.0, help="Constant SFR (Msun/yr)")
    pop.add_argument("--duration-myr", type=float, default=20.0, help="Constant-SFR duration (Myr)")
    pop.add_argument("--n-sims", type=int, default=10_000, help="Monte Carlo realizations")
    pop.add_argument("--seed", type=int, default=1)
    pop.add_argument("--radius-kpc", type=float, default=1.0)
    pop.add_argument("--sun-x", type=float, default=-8.2)
    pop.add_argument("--sun-y", type=float, default=0.0)
    pop.add_argument("--within-samples", type=int, default=200_000)
    pop.add_argument("--fast", action="store_true", help="Binomial phase sampling (recommended)")
    pop.add_argument("--no-fast", dest="fast", action="store_false")
    pop.set_defaults(fast=True)
    pop.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path")
    pop.add_argument("--out-plot", type=Path, default=None, help="Optional PNG summary plot path")
    pop.set_defaults(func=_cmd_population)

    scan = sub.add_parser("imf-scan", help="Compare expected C-burning RSG counts across multiple IMFs.")
    scan.add_argument("--phases-csv", type=Path, required=True)
    scan.add_argument(
        "--imf",
        action="append",
        default=[],
        choices=["kroupa", "salpeter", "top-heavy", "top-light"],
        help="Repeatable; if omitted, scans all presets.",
    )
    scan.add_argument("--sfr", type=float, default=2.0, help="Constant SFR (Msun/yr)")
    scan.add_argument("--duration-myr", type=float, default=20.0, help="Constant-SFR duration (Myr)")
    scan.add_argument("--n-sims", type=int, default=10_000, help="Monte Carlo realizations")
    scan.add_argument("--seed", type=int, default=1)
    scan.add_argument("--radius-kpc", type=float, default=1.0)
    scan.add_argument("--sun-x", type=float, default=-8.2)
    scan.add_argument("--sun-y", type=float, default=0.0)
    scan.add_argument("--within-samples", type=int, default=200_000)
    scan.add_argument("--fast", action="store_true", help="Binomial phase sampling (recommended)")
    scan.add_argument("--no-fast", dest="fast", action="store_false")
    scan.set_defaults(fast=True)
    scan.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path")
    scan.add_argument("--out-plot", type=Path, default=None, help="Optional PNG plot path")
    scan.set_defaults(func=_cmd_imf_scan)

    lm = sub.add_parser("list-masses", help="List initial masses available in a PARSEC track ZIP.")
    lm.add_argument("--zip", type=Path, required=True)
    lm.set_defaults(func=_cmd_list_masses)

    phr = sub.add_parser("plot-hr", help="Plot a PARSEC track HR diagram and highlight RSG/C-burning regions.")
    phr.add_argument("--zip", type=Path, required=True)
    phr.add_argument("--mass", type=float, required=True)
    phr.add_argument("--out", type=Path, required=True)
    phr.add_argument("--lc-threshold", type=float, default=1e-3)
    phr.add_argument("--rsg-teff-max-k", type=float, default=4000.0)
    phr.add_argument("--rsg-logl-min", type=float, default=4.5)
    phr.set_defaults(func=_cmd_plot_hr)

    ps = sub.add_parser("plot-spectrum", help="Plot the toy alpha-fit neutrino spectrum.")
    ps.add_argument("--mean-energy-mev", type=float, default=0.9)
    ps.add_argument("--alpha", type=float, default=2.0)
    ps.add_argument("--out", type=Path, required=True)
    ps.add_argument("--e-min-mev", type=float, default=0.0)
    ps.add_argument("--e-max-mev", type=float, default=5.0)
    ps.add_argument("--n-grid", type=int, default=2000)
    ps.set_defaults(func=_cmd_plot_spectrum)

    ev = sub.add_parser("events", help="Estimate neutrino-electron ES event rate for a flux + spectrum.")
    ev.add_argument("--flux-cm2-s", type=float, required=True, help="Total neutrino flux at Earth (cm^-2 s^-1)")
    ev.add_argument("--mean-energy-mev", type=float, default=0.9, help="Spectrum mean energy <E> (MeV)")
    ev.add_argument("--alpha", type=float, default=2.0, help="Pinching parameter for alpha-fit spectrum")
    ev.add_argument("--flavor", choices=["nue", "numu", "nutau"], default="nue")
    ev.add_argument("--detector-kton", type=float, default=22.5, help="Fiducial mass (kton water-equivalent)")
    ev.add_argument("--detector-name", type=str, default="SK-like")
    ev.add_argument("--e-min-mev", type=float, default=0.1)
    ev.add_argument("--e-max-mev", type=float, default=5.0)
    ev.add_argument("--n-grid", type=int, default=2000, help="Energy grid points for integration")
    ev.set_defaults(func=_cmd_events)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))

