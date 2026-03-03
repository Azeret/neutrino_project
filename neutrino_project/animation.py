from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .galaxy import MWYoungSFParams, sample_mw_young_xy
from .imf import TwoPartIMF, imf_preset


@dataclass(frozen=True)
class PhaseWindows:
    """
    Phase windows for one representative mass point (from PARSEC-derived CSV).

    Times are "age since birth / ZAMS" in Myr.
    """

    lifetime_myr: float
    rsg_start_myr: float
    rsg_end_myr: float
    cburn_start_myr: float | None
    cburn_end_myr: float | None


@dataclass(frozen=True)
class ParticlePopulation:
    """
    A simple particle-based population for visualization/animation.

    Each particle represents one star formed in a constant-SFR window.
    Positions are fixed (no kinematics).
    """

    x_kpc: np.ndarray
    y_kpc: np.ndarray
    birth_myr: np.ndarray
    mass_msun: np.ndarray

    lifetime_myr: np.ndarray
    rsg_start_myr: np.ndarray
    rsg_end_myr: np.ndarray
    cburn_start_myr: np.ndarray  # NaN if not defined
    cburn_end_myr: np.ndarray  # NaN if not defined

    @property
    def n(self) -> int:
        return int(self.x_kpc.size)


def _powerlaw_sample(rng: np.random.Generator, n: int, m0: float, m1: float, alpha: float) -> np.ndarray:
    """
    Sample m in [m0,m1] with PDF ∝ m^{-alpha}.
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    if m1 <= m0:
        raise ValueError("m1 must be > m0")
    a = float(alpha)
    u = rng.random(int(n))
    if abs(a - 1.0) < 1e-12:
        return m0 * (m1 / m0) ** u
    p = 1.0 - a
    return (m0**p + u * (m1**p - m0**p)) ** (1.0 / p)


def sample_masses_from_imf(
    rng: np.random.Generator,
    imf: TwoPartIMF,
    n: int,
    m0: float,
    m1: float,
) -> np.ndarray:
    """
    Sample masses from the TwoPartIMF within [m0,m1].

    This is used only for visualization; the main pipeline integrates the IMF
    analytically for expected counts.
    """
    m0 = float(max(m0, imf.m_min))
    m1 = float(min(m1, imf.m_max))
    if m1 <= m0:
        return np.empty(0, dtype=float)
    if n <= 0:
        return np.empty(0, dtype=float)

    if m1 <= imf.m_break:
        return _powerlaw_sample(rng, n, m0, m1, imf.alpha1)
    if m0 >= imf.m_break:
        return _powerlaw_sample(rng, n, m0, m1, imf.alpha2)

    # Crosses the break: split by expected numbers on each side.
    # Use the same integrals as TwoPartIMF.number_per_msun(), but only for relative weights.
    def _int_num(lo: float, hi: float, alpha: float) -> float:
        if hi <= lo:
            return 0.0
        if abs(alpha - 1.0) < 1e-12:
            return float(np.log(hi / lo))
        p = 1.0 - alpha
        return float((hi**p - lo**p) / p)

    k2_over_k1 = imf.m_break ** (imf.alpha2 - imf.alpha1)
    n_low = _int_num(m0, imf.m_break, imf.alpha1)
    n_high = k2_over_k1 * _int_num(imf.m_break, m1, imf.alpha2)
    if (n_low + n_high) <= 0:
        return np.empty(0, dtype=float)

    frac_low = n_low / (n_low + n_high)
    n_low_s = rng.binomial(int(n), frac_low)
    n_high_s = int(n) - int(n_low_s)
    a = _powerlaw_sample(rng, n_low_s, m0, imf.m_break, imf.alpha1)
    b = _powerlaw_sample(rng, n_high_s, imf.m_break, m1, imf.alpha2)
    out = np.concatenate([a, b])
    rng.shuffle(out)
    return out


def load_phase_windows_by_massbin(phases_csv: Path | str) -> list[tuple[float, float, PhaseWindows]]:
    """
    Load PARSEC-derived phase windows from the shipped CSV and convert them into
    per-mass-bin windows, using the same "mass-bin edges" convention as the main population code.

    Returns a list of tuples: (m_lo, m_hi, PhaseWindows)
    """
    path = Path(phases_csv)
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"{path} is empty")

    masses = np.array([float(r["mass_msun"]) for r in rows], dtype=float)
    order = np.argsort(masses)
    masses = masses[order]
    rows = [rows[i] for i in order]
    if masses.size < 2:
        raise RuntimeError("Need at least 2 masses in phases CSV to build bins")

    edges: list[float] = []
    for i, m in enumerate(masses.tolist()):
        if i == 0:
            lo = m - 0.5 * (masses[i + 1] - m)
            edges.append(max(0.1, float(lo)))
        hi = m + 0.5 * (masses[i + 1] - m) if i < masses.size - 1 else m + 0.5 * (m - masses[i - 1])
        edges.append(float(hi))

    out: list[tuple[float, float, PhaseWindows]] = []
    for i, row in enumerate(rows):
        lifetime = float(row["lifetime_myr"])
        rsg_dur = float(row["rsg_duration_myr"])
        rsg_end = lifetime
        rsg_start = max(0.0, rsg_end - rsg_dur)
        cb0 = row.get("cburn_start_myr", "").strip()
        cb1 = row.get("cburn_end_myr", "").strip()
        cburn_start = float(cb0) if cb0 else None
        cburn_end = float(cb1) if cb1 else None

        out.append(
            (
                float(edges[i]),
                float(edges[i + 1]),
                PhaseWindows(
                    lifetime_myr=lifetime,
                    rsg_start_myr=rsg_start,
                    rsg_end_myr=rsg_end,
                    cburn_start_myr=cburn_start,
                    cburn_end_myr=cburn_end,
                ),
            )
        )
    return out


def simulate_particles_for_animation(
    *,
    phases_csv: Path | str,
    imf: str,
    sfr_msun_per_yr: float,
    t_max_myr: float,
    seed: int = 1,
    spatial_params: MWYoungSFParams | None = None,
    m_min: float = 12.0,
    m_max: float = 35.0,
) -> ParticlePopulation:
    """
    Simulate a particle population for a MW 2D animation.

    This intentionally only simulates the massive-star range covered by the PARSEC phase windows
    (default 12–35 Msun), so that RSG/C-burning phases are defined for every particle.
    """
    rng = np.random.default_rng(seed)
    imf_model = imf_preset(imf)
    bins = load_phase_windows_by_massbin(phases_csv)

    formed_mass_msun = float(sfr_msun_per_yr) * float(t_max_myr) * 1e6
    stars: list[tuple[np.ndarray, PhaseWindows]] = []

    for m_lo, m_hi, w in bins:
        lo = max(float(m_lo), float(m_min))
        hi = min(float(m_hi), float(m_max))
        if hi <= lo:
            continue
        mu = formed_mass_msun * imf_model.number_per_msun(lo, hi)
        n = int(rng.poisson(mu))
        if n <= 0:
            continue
        masses = sample_masses_from_imf(rng, imf_model, n, lo, hi)
        stars.append((masses, w))

    if not stars:
        raise RuntimeError("No particles were generated; check mass range and phases CSV.")

    mass = np.concatenate([m for m, _ in stars]).astype(float, copy=False)
    birth = rng.uniform(0.0, float(t_max_myr), mass.size).astype(float, copy=False)

    lifetime = np.empty_like(mass)
    rsg_start = np.empty_like(mass)
    rsg_end = np.empty_like(mass)
    cb0 = np.empty_like(mass)
    cb1 = np.empty_like(mass)

    idx = 0
    for masses, w in stars:
        n = masses.size
        lifetime[idx : idx + n] = w.lifetime_myr
        rsg_start[idx : idx + n] = w.rsg_start_myr
        rsg_end[idx : idx + n] = w.rsg_end_myr
        cb0[idx : idx + n] = np.nan if w.cburn_start_myr is None else float(w.cburn_start_myr)
        cb1[idx : idx + n] = np.nan if w.cburn_end_myr is None else float(w.cburn_end_myr)
        idx += n

    if spatial_params is None:
        spatial_params = MWYoungSFParams()
    x, y = sample_mw_young_xy(mass.size, rng, spatial_params)

    return ParticlePopulation(
        x_kpc=x.astype(float, copy=False),
        y_kpc=y.astype(float, copy=False),
        birth_myr=birth,
        mass_msun=mass,
        lifetime_myr=lifetime,
        rsg_start_myr=rsg_start,
        rsg_end_myr=rsg_end,
        cburn_start_myr=cb0,
        cburn_end_myr=cb1,
    )


def render_mw_evolution_frames(
    *,
    pop: ParticlePopulation,
    out_dir: Path,
    t_grid_myr: np.ndarray,
    sun_xy_kpc: tuple[float, float] = (-8.2, 0.0),
    radius_kpc: float = 1.0,
    max_points: int = 60_000,
    seed: int = 1,
) -> None:
    """
    Render a sequence of PNG frames for the evolving MW population map.

    Colors:
    - grey: alive but not RSG/C-burning
    - red: RSG
    - blue: C-burning
    - purple: overlap (C-burning RSG)
    - black: dead (shown as 'black holes' / remnants)
    """
    if t_grid_myr.ndim != 1 or t_grid_myr.size < 2:
        raise ValueError("t_grid_myr must be a 1D array with at least 2 entries")

    rng = np.random.default_rng(seed + 2026)
    idx = np.arange(pop.n)
    if idx.size > int(max_points):
        idx = rng.choice(idx, size=int(max_points), replace=False)

    x = pop.x_kpc[idx]
    y = pop.y_kpc[idx]
    birth = pop.birth_myr[idx]
    lifetime = pop.lifetime_myr[idx]
    rsg_start = pop.rsg_start_myr[idx]
    cb0 = pop.cburn_start_myr[idx]
    cb1 = pop.cburn_end_myr[idx]

    from .plots import _ensure_matplotlib_cache_dirs

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sx, sy = sun_xy_kpc

    for k, t in enumerate(t_grid_myr.tolist()):
        age = float(t) - birth
        born = age >= 0.0
        alive = born & (age < lifetime)
        dead = born & ~alive

        rsg = alive & (age >= rsg_start)
        cburn = alive & np.isfinite(cb0) & np.isfinite(cb1) & (age >= cb0) & (age <= cb1)
        overlap = rsg & cburn
        rsg_only = rsg & ~overlap
        cburn_only = cburn & ~overlap
        other_alive = alive & ~(rsg | cburn)

        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        ax.scatter(x[dead], y[dead], s=6, color="black", alpha=0.45, label="dead (BH/remnant)", rasterized=True)
        ax.scatter(x[other_alive], y[other_alive], s=6, color="0.6", alpha=0.30, label="alive (other)", rasterized=True)
        ax.scatter(x[rsg_only], y[rsg_only], s=10, color="#d62728", alpha=0.85, label="RSG", rasterized=True)
        ax.scatter(x[cburn_only], y[cburn_only], s=12, color="#1f77b4", alpha=0.85, label="C-burning", rasterized=True)
        ax.scatter(x[overlap], y[overlap], s=14, color="#9467bd", alpha=0.95, label="C-burning RSG", rasterized=True)

        ax.scatter([0.0], [0.0], s=90, color="black", label="Galactic Center")
        ax.scatter([sx], [sy], s=90, color="cyan", marker="*", label="Sun")
        circ = plt.Circle((sx, sy), radius_kpc, color="cyan", fill=False, lw=2, ls="--", alpha=0.8)
        ax.add_patch(circ)

        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_title(f"Toy MW buildup (massive stars only): t={t:0.2f} Myr")
        ax.text(
            0.02,
            0.02,
            f"N shown={x.size}\n"
            f"alive={int(alive.sum())}\n"
            f"RSG={int(rsg.sum())}\n"
            f"C-burn={int(cburn.sum())}\n"
            f"dead={int(dead.sum())}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.8"),
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")

        fig.tight_layout()
        fig.savefig(out_dir / f"frame_{k:04d}.png", dpi=160)
        plt.close(fig)


def make_video_from_frames(
    *,
    frames_dir: Path,
    out_mp4: Path | None,
    out_gif: Path | None,
    fps: int = 10,
) -> None:
    """
    Turn `frame_0000.png`... into an MP4 and/or GIF using ffmpeg (if available).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    pattern = str(frames_dir / "frame_%04d.png")

    if out_mp4 is not None:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(int(fps)),
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "26",
                str(out_mp4),
            ],
            check=True,
        )

    if out_gif is not None:
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(int(fps)),
                "-i",
                pattern,
                "-vf",
                "scale=720:-1:flags=lanczos",
                str(out_gif),
            ],
            check=True,
        )


def make_mw_evolution_animation(
    *,
    phases_csv: Path | str,
    out_frames_dir: Path,
    out_mp4: Path | None,
    out_gif: Path | None,
    imf: str = "kroupa",
    sfr_msun_per_yr: float = 2.0,
    t_max_myr: float = 20.0,
    dt_myr: float = 0.5,
    fps: int = 10,
    max_points: int = 60_000,
    seed: int = 1,
    sun_xy_kpc: tuple[float, float] = (-8.2, 0.0),
    radius_kpc: float = 1.0,
) -> None:
    """
    High-level helper: simulate particles, render frames, and optionally build MP4/GIF.
    """
    pop = simulate_particles_for_animation(
        phases_csv=phases_csv,
        imf=imf,
        sfr_msun_per_yr=sfr_msun_per_yr,
        t_max_myr=t_max_myr,
        seed=seed,
    )
    t_grid = np.arange(0.0, float(t_max_myr) + 0.5 * float(dt_myr), float(dt_myr))
    render_mw_evolution_frames(
        pop=pop,
        out_dir=out_frames_dir,
        t_grid_myr=t_grid,
        sun_xy_kpc=sun_xy_kpc,
        radius_kpc=radius_kpc,
        max_points=max_points,
        seed=seed,
    )
    if (out_mp4 is not None) or (out_gif is not None):
        if shutil.which("ffmpeg") is None:
            print("Note: ffmpeg not found; skipping MP4/GIF. Frames are still available in:", out_frames_dir)
            return
        make_video_from_frames(frames_dir=out_frames_dir, out_mp4=out_mp4, out_gif=out_gif, fps=fps)
