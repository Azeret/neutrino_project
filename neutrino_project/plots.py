from __future__ import annotations

from pathlib import Path

import numpy as np

from .detectability import detector_presets, mean_inv_d2_within_radius, one_star_flux_cm2_s
from .parsec_v2_vms import load_track_arrays
from .population import expected_counts_vs_time, load_phases_csv, simulate_snapshot_catalog


_MEV_TO_ERG = 1.602176634e-6


def _ensure_matplotlib_cache_dirs() -> None:
    """
    Avoid slow imports + noisy warnings when user home is not writable.
    """
    import os

    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    Path(os.environ.setdefault("MPLCONFIGDIR", str(tmp / "matplotlib"))).mkdir(parents=True, exist_ok=True)
    Path(os.environ.setdefault("XDG_CACHE_HOME", str(tmp / "xdg_cache"))).mkdir(parents=True, exist_ok=True)


def plot_parsec_hrd(
    *,
    zip_path: Path,
    mass_msun: float,
    out_png: Path,
    lc_threshold: float = 1e-3,
    rsg_teff_max_k: float = 4000.0,
    rsg_logl_min: float = 4.5,
) -> None:
    """
    Simple HR diagram (logTe vs logL) and highlight:
    - RSG region (toy criterion)
    - carbon burning (LC threshold)
    - overlap
    """
    track = load_track_arrays(zip_path=zip_path, mass_msun=mass_msun, columns=("LOG_TE", "LOG_L", "LC"))
    log_te = track["LOG_TE"]
    log_l = track["LOG_L"]
    lc = track["LC"]

    teff = 10.0**log_te
    rsg = (teff <= rsg_teff_max_k) & (log_l >= rsg_logl_min)
    cburn = lc >= lc_threshold
    both = rsg & cburn

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(log_te, log_l, color="0.75", lw=1.0, label="track")
    if rsg.any():
        ax.scatter(log_te[rsg], log_l[rsg], s=8, color="#d62728", label="RSG")
    if cburn.any():
        ax.scatter(log_te[cburn], log_l[cburn], s=10, color="#1f77b4", label="C-burning (LC)")
    if both.any():
        ax.scatter(log_te[both], log_l[both], s=14, color="#9467bd", label="overlap")

    ax.invert_xaxis()
    ax.set_xlabel(r"$\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$")
    ax.set_ylabel(r"$\log_{10}(L/L_\odot)$")
    ax.set_title(f"PARSEC v2.0 VMS track: {mass_msun:g} Msun")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_alpha_spectrum(
    *,
    mean_energy_mev: float,
    alpha: float,
    out_png: Path,
    e_min_mev: float = 0.0,
    e_max_mev: float = 5.0,
    n_grid: int = 2000,
) -> None:
    from .neutrinos import NeutrinoSpectrumModel

    spec = NeutrinoSpectrumModel.alpha_fit(mean_energy_mev=mean_energy_mev, alpha=alpha)
    e = np.linspace(e_min_mev, e_max_mev, int(n_grid))
    f = spec.pdf(e)
    if f.max() > 0:
        f = f / np.trapz(f, e)

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(e, f, color="black")
    ax.set_xlabel("Eν [MeV]")
    ax.set_ylabel("f(E) [MeV⁻¹] (normalized)")
    ax.set_title(f"Alpha-fit number spectrum (<E>={mean_energy_mev:g} MeV, α={alpha:g})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_cmd39_isochrone_hrd(
    *,
    isochrone_dat: Path,
    out_png: Path,
    rsg_teff_max_k: float = 4000.0,
    rsg_logl_min: float = 4.5,
) -> None:
    """
    Plot a CMD 3.9 "isochrone tables" file (PARSEC/COLIBRI) in the HR diagram.

    This plot is mainly pedagogical: it shows the *RSG cut* (Teff + logL) on
    an isochrone. The isochrone tables do not label carbon-burning, so C-burning
    cannot be identified from this file alone.
    """
    path = Path(isochrone_dat)
    header_cols: list[str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") and "Zini" in line and "logTe" in line and "logL" in line:
                header_cols = line.lstrip("#").strip().split()
                break
    if header_cols is None:
        raise RuntimeError(f"Could not find CMD header with columns in {path}")

    i_logl = header_cols.index("logL")
    i_logte = header_cols.index("logTe")
    data = np.loadtxt(path, comments="#", usecols=[i_logte, i_logl])
    log_te = data[:, 0]
    log_l = data[:, 1]
    teff_k = 10.0**log_te
    rsg = (teff_k <= rsg_teff_max_k) & (log_l >= rsg_logl_min)

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(log_te, log_l, color="0.7", lw=1.0, label="isochrone")
    if rsg.any():
        ax.scatter(log_te[rsg], log_l[rsg], s=10, color="#d62728", label="RSG cut")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$")
    ax.set_ylabel(r"$\log_{10}(L/L_\odot)$")
    ax.set_title("CMD 3.9 isochrone (HR diagram)\n(RSG cut shown; carbon burning is not labeled in CMD tables)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_phase_timeline(
    *,
    phases_csv: Path,
    mass_msun: float,
    out_png: Path,
) -> None:
    """
    Show how the project estimates phase windows for one mass.

    Uses the phase CSV (so it works even without the full PARSEC track ZIPs).
    """
    pts = load_phases_csv(phases_csv)
    p = next((x for x in pts if abs(x.mass_msun - mass_msun) < 1e-6), None)
    if p is None:
        available = ", ".join(f"{x.mass_msun:g}" for x in pts)
        raise ValueError(f"Mass {mass_msun:g} not found in {phases_csv}. Available: {available}")

    lif = p.lifetime_myr
    rsg0, rsg1 = p.rsg_start_myr, p.rsg_end_myr
    if p.cburn_start_myr is not None and p.cburn_end_myr is not None:
        cb0, cb1 = p.cburn_start_myr, p.cburn_end_myr
    else:
        cb1 = lif
        cb0 = max(0.0, cb1 - p.cburn_duration_myr)

    ov0 = max(rsg0, cb0)
    ov1 = min(rsg1, cb1)

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Carbon burning can be extremely short (kyr) compared to the full lifetime (Myr),
    # so we show both a full-lifetime view and a zoom near the end of life.
    zoom_myr = max(0.02, 20.0 * max(p.cburn_duration_myr, p.rsg_duration_myr, 1e-4))
    zoom_lo = max(0.0, lif - zoom_myr)
    zoom_hi = lif

    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(7.6, 4.8),
        gridspec_kw={"height_ratios": [1.0, 1.2]},
        sharey=True,
    )

    for ax in (ax0, ax1):
        ax.set_yticks([])
        ax.hlines(0.8, 0, lif, color="0.85", lw=6, label="alive range" if ax is ax0 else None)
        ax.hlines(0.6, rsg0, rsg1, color="#d62728", lw=10, label="RSG" if ax is ax0 else None)
        ax.hlines(0.4, cb0, cb1, color="#1f77b4", lw=10, label="C-burning (LC)" if ax is ax0 else None)
        if ov1 > ov0:
            ax.hlines(0.2, ov0, ov1, color="#9467bd", lw=10, label="overlap" if ax is ax0 else None)

        # Make the tiny windows visible in the zoom by marking boundaries.
        ax.vlines([cb0, cb1], 0.32, 0.48, color="#1f77b4", lw=1.5)
        ax.vlines([rsg0, rsg1], 0.52, 0.68, color="#d62728", lw=1.5)
        if ov1 > ov0:
            ax.vlines([ov0, ov1], 0.12, 0.28, color="#9467bd", lw=1.5)

    ax0.set_title(f"Phase windows for {mass_msun:g} Msun (toy definitions)")
    ax0.set_xlim(0, lif * 1.02)
    ax0.set_xlabel("Age since birth [Myr] (full lifetime)")
    ax0.set_ylim(0, 1.05)
    ax0.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25))

    ax1.set_xlim(zoom_lo, zoom_hi)
    ax1.set_xlabel(f"Zoom near end of life (last {zoom_hi - zoom_lo:.3g} Myr)")
    ax1.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_mw_snapshot_map(
    *,
    phases_csv: Path,
    out_png: Path,
    imf: str = "kroupa",
    sfr_msun_per_yr: float = 2.0,
    t_obs_myr: float = 20.0,
    seed: int = 1,
    sun_xy_kpc: tuple[float, float] = (-8.2, 0.0),
    radius_kpc: float = 1.0,
    max_points: int = 50_000,
) -> None:
    """
    2D Milky Way disk snapshot (toy), with colors:
    - grey: alive but not RSG/C-burning
    - red: RSG
    - blue: C-burning
    - purple: overlap (C-burning RSG)
    - light grey: dead (formed in the time window but already ended its lifetime)
    """
    cat = simulate_snapshot_catalog(
        phases_csv=phases_csv,
        imf=imf,
        sfr_msun_per_yr=sfr_msun_per_yr,
        t_obs_myr=t_obs_myr,
        seed=seed,
    )
    x, y = cat.x_kpc, cat.y_kpc
    alive = cat.alive
    rsg = cat.is_rsg
    cburn = cat.is_cburn
    overlap = rsg & cburn
    dead = ~alive
    other_alive = alive & ~(rsg | cburn)
    rsg_only = rsg & ~overlap
    cburn_only = cburn & ~overlap

    rng = np.random.default_rng(seed + 12345)
    idx = np.arange(x.size)
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
        x, y = x[idx], y[idx]
        dead = dead[idx]
        other_alive = other_alive[idx]
        rsg_only = rsg_only[idx]
        cburn_only = cburn_only[idx]
        overlap = overlap[idx]

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    ax.scatter(x[dead], y[dead], s=4, color="0.9", alpha=0.35, label="dead (in window)")
    ax.scatter(x[other_alive], y[other_alive], s=6, color="0.6", alpha=0.35, label="alive (other)")
    ax.scatter(x[rsg_only], y[rsg_only], s=10, color="#d62728", alpha=0.8, label="RSG")
    ax.scatter(x[cburn_only], y[cburn_only], s=12, color="#1f77b4", alpha=0.8, label="C-burning")
    ax.scatter(x[overlap], y[overlap], s=14, color="#9467bd", alpha=0.9, label="C-burning RSG")

    sx, sy = sun_xy_kpc
    ax.scatter([0.0], [0.0], s=90, color="black", label="Galactic Center")
    ax.scatter([sx], [sy], s=90, color="cyan", marker="*", label="Sun")
    circ = plt.Circle((sx, sy), radius_kpc, color="cyan", fill=False, lw=2, ls="--", alpha=0.8)
    ax.add_patch(circ)

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"Toy MW snapshot at t={t_obs_myr:g} Myr (IMF={imf})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_counts_within_radius_vs_time(
    *,
    phases_csv: Path,
    out_png: Path,
    imf: str = "kroupa",
    sfr_msun_per_yr: float = 2.0,
    t_max_myr: float = 25.0,
    radius_kpc: float = 1.0,
    sun_xy_kpc: tuple[float, float] = (-8.2, 0.0),
    within_samples: int = 200_000,
    seed: int = 1,
) -> None:
    """
    Expected number of RSGs and C-burning RSGs within a radius around the Sun, vs time.
    """
    t = np.linspace(0.0, t_max_myr, 220)
    d = expected_counts_vs_time(
        phases_csv=phases_csv,
        imf=imf,
        sfr_msun_per_yr=sfr_msun_per_yr,
        t_grid_myr=t,
        radius_kpc=radius_kpc,
        sun_x_kpc=sun_xy_kpc[0],
        sun_y_kpc=sun_xy_kpc[1],
        within_samples=within_samples,
        seed=seed,
    )
    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True)

    ax0.plot(d["t_myr"], d["rsg"], color="#d62728", lw=2, label="RSG (whole MW)")
    ax0.plot(d["t_myr"], d["cburn_rsg"], color="#9467bd", lw=2, label="C-burning RSG (whole MW)")
    ax0.set_ylabel("Expected number (whole MW)")
    ax0.legend(frameon=False)

    ax1.plot(d["t_myr"], d["rsg_within"], color="#d62728", lw=2, label=f"RSG within {radius_kpc:g} kpc")
    ax1.plot(
        d["t_myr"],
        d["cburn_rsg_within"],
        color="#9467bd",
        lw=2,
        label=f"C-burning RSG within {radius_kpc:g} kpc",
    )
    ax1.set_xlabel("Time since start of constant SFR [Myr]")
    ax1.set_ylabel("Expected number (local)")
    ax1.legend(frameon=False)

    fig.suptitle(f"Toy MW counts vs time (IMF={imf}, SFR={sfr_msun_per_yr:g} Msun/yr)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _mean_inv_d2_within_radius(
    *,
    n_samples: int,
    radius_kpc: float,
    sun_xy_kpc: tuple[float, float],
    seed: int,
) -> float:
    # Backward-compatible wrapper (this used to live here).
    return mean_inv_d2_within_radius(
        n_samples=n_samples,
        radius_kpc=radius_kpc,
        sun_xy_kpc=sun_xy_kpc,
        seed=seed,
    )


def plot_neutrino_yield_vs_time(
    *,
    phases_csv: Path,
    out_png: Path,
    imf: str = "kroupa",
    sfr_msun_per_yr: float = 2.0,
    t_max_myr: float = 25.0,
    radius_kpc: float = 1.0,
    sun_xy_kpc: tuple[float, float] = (-8.2, 0.0),
    within_samples: int = 200_000,
    seed: int = 1,
    lnu_per_star_erg_s: float = 1e38,
    mean_energy_mev: float = 0.9,
    alpha: float = 2.0,
    detector_kton: float = 22.5,
) -> None:
    """
    Toy time evolution of the *expected* neutrino flux at Earth.

    Assumptions:
    - each C-burning RSG has the same neutrino luminosity L_nu (CONFIG parameter)
    - we only include stars within `radius_kpc` of the Sun
    - flux is estimated using <1/d^2> over the toy spatial distribution within the radius
    """
    from .neutrinos import Detector, NeutrinoSpectrumModel, estimate_es_events_per_year

    t = np.linspace(0.0, t_max_myr, 220)
    d = expected_counts_vs_time(
        phases_csv=phases_csv,
        imf=imf,
        sfr_msun_per_yr=sfr_msun_per_yr,
        t_grid_myr=t,
        radius_kpc=radius_kpc,
        sun_x_kpc=sun_xy_kpc[0],
        sun_y_kpc=sun_xy_kpc[1],
        within_samples=within_samples,
        seed=seed,
    )
    n_cburn_rsg_within = d["cburn_rsg_within"]

    inv_d2 = mean_inv_d2_within_radius(
        n_samples=max(50_000, within_samples // 2),
        radius_kpc=radius_kpc,
        sun_xy_kpc=sun_xy_kpc,
        seed=seed + 999,
    )
    flux = n_cburn_rsg_within * one_star_flux_cm2_s(
        lnu_erg_s=lnu_per_star_erg_s,
        mean_energy_mev=mean_energy_mev,
        inv_d2_cm2=inv_d2,
    )

    spec = NeutrinoSpectrumModel.alpha_fit(mean_energy_mev=mean_energy_mev, alpha=alpha)

    # Compare a few detector scales (toy, ES-only).
    detectors = detector_presets()
    # Override the SK-like mass from the function argument while keeping the same label format.
    detectors = [
        detectors[0].__class__(name=f"SK-like ({detector_kton:g} kt)", fiducial_mass_kton=float(detector_kton)),
        *detectors[1:],
    ]
    events_by_det = {
        det.to_detector().name: np.array(
            [
                estimate_es_events_per_year(
                    flux_at_earth_cm2_s=float(f),
                    detector=det.to_detector(),
                    spectrum=spec,
                )
                for f in flux
            ]
        )
        for det in detectors
    }

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(t, flux, color="black", lw=2, label="expected flux (within radius)")
    ax1.set_xlabel("Time since start of constant SFR [Myr]")
    ax1.set_ylabel(r"Flux at Earth [cm$^{-2}$ s$^{-1}$]")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, ev), c in zip(events_by_det.items(), colors, strict=False):
        ax2.plot(t, ev, color=c, lw=2, label=f"{name}: ES events/year")
    ax2.axhline(1.0, color="0.5", lw=1.0, ls="--")
    ax2.text(t_max_myr * 0.02, 1.2, "≈1 event/yr", color="0.4", fontsize=8)
    ax2.set_ylabel("Toy ES events/year")
    ax2.set_yscale("log")

    ax1.set_title(
        f"Toy neutrino yield vs time (IMF={imf}, radius={radius_kpc:g} kpc)\n"
        f"(ES-only; backgrounds/thresholds not included)"
    )

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, frameon=False, fontsize=7.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
