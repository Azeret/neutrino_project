from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .galaxy import MWYoungSFParams, estimate_within_probability, sample_mw_young_xy
from .imf import imf_preset


@dataclass(frozen=True)
class PhasePoint:
    mass_msun: float
    lifetime_myr: float
    cburn_start_myr: float | None
    cburn_end_myr: float | None
    cburn_duration_myr: float
    rsg_duration_myr: float
    cburn_rsg_duration_myr: float

    @property
    def rsg_end_myr(self) -> float:
        # In this toy model we approximate RSG as occurring late in the evolution.
        return self.lifetime_myr

    @property
    def rsg_start_myr(self) -> float:
        return max(0.0, self.rsg_end_myr - self.rsg_duration_myr)


def load_phases_csv(path: Path | str) -> list[PhasePoint]:
    path = Path(path)
    def _opt_float(row: dict[str, str], key: str) -> float | None:
        if key not in row:
            return None
        v = (row.get(key) or "").strip()
        if v == "":
            return None
        return float(v)

    out: list[PhasePoint] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        need = {
            "mass_msun",
            "lifetime_myr",
            "cburn_duration_kyr",
            "rsg_duration_myr",
            "cburn_rsg_duration_kyr",
        }
        missing = need - set(r.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
        for row in r:
            out.append(
                PhasePoint(
                    mass_msun=float(row["mass_msun"]),
                    lifetime_myr=float(row["lifetime_myr"]),
                    cburn_start_myr=_opt_float(row, "cburn_start_myr"),
                    cburn_end_myr=_opt_float(row, "cburn_end_myr"),
                    cburn_duration_myr=float(row["cburn_duration_kyr"]) / 1e3,
                    rsg_duration_myr=float(row["rsg_duration_myr"]),
                    cburn_rsg_duration_myr=float(row["cburn_rsg_duration_kyr"]) / 1e3,
                )
            )
    if not out:
        raise RuntimeError(f"{path} is empty")
    out.sort(key=lambda p: p.mass_msun)
    return out


@dataclass(frozen=True)
class MassBin:
    m_center: float
    m_lo: float
    m_hi: float

    lifetime_myr: float
    cburn_start_myr: float | None
    cburn_end_myr: float | None
    cburn_duration_myr: float
    rsg_duration_myr: float
    cburn_rsg_duration_myr: float


def make_mass_bins(points: list[PhasePoint]) -> list[MassBin]:
    masses = [p.mass_msun for p in points]
    if len(masses) < 2:
        raise ValueError("Need at least 2 masses to build bins")

    edges: list[float] = []
    for i, m in enumerate(masses):
        if i == 0:
            lo = m - 0.5 * (masses[i + 1] - m)
            edges.append(max(0.1, lo))
        hi = m + 0.5 * (masses[i + 1] - m) if i < len(masses) - 1 else m + 0.5 * (m - masses[i - 1])
        edges.append(hi)

    bins: list[MassBin] = []
    for i, p in enumerate(points):
        bins.append(
            MassBin(
                m_center=p.mass_msun,
                m_lo=edges[i],
                m_hi=edges[i + 1],
                lifetime_myr=p.lifetime_myr,
                cburn_start_myr=p.cburn_start_myr,
                cburn_end_myr=p.cburn_end_myr,
                cburn_duration_myr=p.cburn_duration_myr,
                rsg_duration_myr=p.rsg_duration_myr,
                cburn_rsg_duration_myr=p.cburn_rsg_duration_myr,
            )
        )
    return bins


def _make_mass_bins(points: list[PhasePoint]) -> list[MassBin]:
    # Backward-compatible alias (internal).
    return make_mass_bins(points)


def _interval_len_in_range(start: float, end: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    a = max(start, lo)
    b = min(end, hi)
    return max(0.0, b - a)


def expected_counts_vs_time(
    *,
    phases_csv: Path | str,
    imf: str,
    sfr_msun_per_yr: float,
    t_grid_myr: np.ndarray,
    radius_kpc: float = 1.0,
    sun_x_kpc: float = -8.2,
    sun_y_kpc: float = 0.0,
    within_samples: int = 200_000,
    seed: int = 1,
    p_within_override: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Deterministic expectation values vs time for a constant-SFR toy model.

    Interpretation:
    - star formation starts at t=0 and continues at constant SFR until t
    - we observe the Galaxy at the same time t
    """
    points = load_phases_csv(phases_csv)
    bins = make_mass_bins(points)
    imf_obj = imf_preset(imf)

    if p_within_override is None:
        rng = np.random.default_rng(seed)
        spatial = MWYoungSFParams()
        p_within = estimate_within_probability(
            rng=rng,
            n_samples=within_samples,
            spatial_params=spatial,
            sun_xy_kpc=(sun_x_kpc, sun_y_kpc),
            radius_kpc=radius_kpc,
        )
    else:
        p_within = float(p_within_override)
        if not (0.0 <= p_within <= 1.0):
            raise ValueError("p_within_override must be in [0,1]")

    t = np.asarray(t_grid_myr, dtype=float)
    if np.any(t < 0):
        raise ValueError("t_grid_myr must be non-negative")

    alive = np.zeros_like(t)
    rsg = np.zeros_like(t)
    cburn = np.zeros_like(t)
    cburn_rsg = np.zeros_like(t)

    for b in bins:
        births_per_myr = sfr_msun_per_yr * 1e6 * imf_obj.number_per_msun(b.m_lo, b.m_hi)
        if births_per_myr <= 0:
            continue

        eff = np.minimum(t, b.lifetime_myr)
        alive += births_per_myr * eff

        rsg_start = max(0.0, b.lifetime_myr - b.rsg_duration_myr)
        rsg_end = b.lifetime_myr
        rsg += births_per_myr * np.array([_interval_len_in_range(rsg_start, rsg_end, 0.0, e) for e in eff])

        if b.cburn_start_myr is not None and b.cburn_end_myr is not None:
            cb0, cb1 = b.cburn_start_myr, b.cburn_end_myr
        else:
            cb1 = b.lifetime_myr
            cb0 = max(0.0, cb1 - b.cburn_duration_myr)
        cburn += births_per_myr * np.array([_interval_len_in_range(cb0, cb1, 0.0, e) for e in eff])

        ov0 = max(cb0, rsg_start)
        ov1 = min(cb1, rsg_end)
        if ov1 > ov0:
            cburn_rsg += births_per_myr * np.array([_interval_len_in_range(ov0, ov1, 0.0, e) for e in eff])

    within = alive * p_within
    rsg_within = rsg * p_within
    cburn_within = cburn * p_within
    cburn_rsg_within = cburn_rsg * p_within
    return {
        "t_myr": t,
        "p_within": np.array([p_within]),
        "alive": alive,
        "rsg": rsg,
        "cburn": cburn,
        "cburn_rsg": cburn_rsg,
        "within": within,
        "rsg_within": rsg_within,
        "cburn_within": cburn_within,
        "cburn_rsg_within": cburn_rsg_within,
    }

@dataclass(frozen=True)
class PopulationInputs:
    phases_csv: Path
    imf: str = "kroupa"

    sfr_msun_per_yr: float = 2.0
    duration_myr: float = 20.0

    n_sims: int = 10_000
    seed: int = 1

    radius_kpc: float = 1.0
    sun_x_kpc: float = -8.2
    sun_y_kpc: float = 0.0
    within_samples: int = 200_000

    fast: bool = True


@dataclass(frozen=True)
class PopulationSummary:
    p_within: float
    mean: dict[str, float]
    std: dict[str, float]


@dataclass(frozen=True)
class IMFScanRow:
    imf: str
    p_within: float
    mean: dict[str, float]
    std: dict[str, float]
    ratios: dict[str, float]


@dataclass(frozen=True)
class SnapshotCatalog:
    """
    A Monte Carlo sample of stars formed over a constant-SFR window and observed at one time.

    Arrays are aligned by index. Coordinates are in the Galactic plane (kpc).
    """

    x_kpc: np.ndarray
    y_kpc: np.ndarray
    mass_msun: np.ndarray
    age_myr: np.ndarray
    alive: np.ndarray
    is_rsg: np.ndarray
    is_cburn: np.ndarray


def _simulate_once(
    *,
    rng: np.random.Generator,
    bins: list[MassBin],
    imf_name: str,
    sfr_msun_per_yr: float,
    duration_myr: float,
    p_within: float,
    fast: bool,
) -> dict[str, int]:
    imf = imf_preset(imf_name)

    n_alive = 0
    n_rsg = 0
    n_cburn = 0
    n_cburn_rsg = 0

    for b in bins:
        births_per_msun = imf.number_per_msun(b.m_lo, b.m_hi)
        eff_myr = min(duration_myr, b.lifetime_myr)
        mean_alive = sfr_msun_per_yr * (eff_myr * 1e6) * births_per_msun
        n = int(rng.poisson(mean_alive)) if mean_alive > 0 else 0
        if n <= 0:
            continue
        n_alive += n

        if fast:
            p_rsg = 0.0 if eff_myr <= 0 else min(1.0, b.rsg_duration_myr / eff_myr)
            p_cburn = 0.0 if eff_myr <= 0 else min(1.0, b.cburn_duration_myr / eff_myr)
            p_cburn_rsg = 0.0 if eff_myr <= 0 else min(1.0, b.cburn_rsg_duration_myr / eff_myr)
            n_rsg += int(rng.binomial(n, p_rsg))
            n_cburn += int(rng.binomial(n, p_cburn))
            n_cburn_rsg += int(rng.binomial(n, p_cburn_rsg))
        else:
            ages_myr = rng.uniform(0.0, eff_myr, size=n)
            p_rsg = 0.0 if eff_myr <= 0 else min(1.0, b.rsg_duration_myr / eff_myr)
            p_cburn = 0.0 if eff_myr <= 0 else min(1.0, b.cburn_duration_myr / eff_myr)
            p_cburn_rsg = 0.0 if eff_myr <= 0 else min(1.0, b.cburn_rsg_duration_myr / eff_myr)
            n_rsg += int((ages_myr <= p_rsg * eff_myr).sum())
            n_cburn += int((ages_myr <= p_cburn * eff_myr).sum())
            n_cburn_rsg += int((ages_myr <= p_cburn_rsg * eff_myr).sum())

    if n_alive == 0:
        return {
            "alive": 0,
            "rsg": 0,
            "cburn": 0,
            "cburn_rsg": 0,
            "within": 0,
            "cburn_rsg_within": 0,
        }

    n_within = int(rng.binomial(n_alive, p_within))
    n_cburn_rsg_within = int(rng.binomial(n_cburn_rsg, p_within)) if n_cburn_rsg > 0 else 0
    return {
        "alive": n_alive,
        "rsg": n_rsg,
        "cburn": n_cburn,
        "cburn_rsg": n_cburn_rsg,
        "within": n_within,
        "cburn_rsg_within": n_cburn_rsg_within,
    }


def run_population_mc(
    inputs: PopulationInputs,
    *,
    out_json: Path | None = None,
    out_plot: Path | None = None,
) -> PopulationSummary:
    """
    Toy MW population synthesis:
    - constant SFR for a fixed duration
    - IMF gives birth counts per formed mass
    - phase occupancy is estimated from PARSEC-derived phase durations

    This is meant for simple IMF sensitivity studies (expected number of
    C-burning RSGs), not for reproducing the detailed MESA-based neutrino model
    in neutrinos.pdf.
    """
    points = load_phases_csv(inputs.phases_csv)
    bins = _make_mass_bins(points)

    rng = np.random.default_rng(inputs.seed)
    spatial = MWYoungSFParams()
    sun_xy = (inputs.sun_x_kpc, inputs.sun_y_kpc)
    p_within = estimate_within_probability(
        rng=rng,
        n_samples=inputs.within_samples,
        spatial_params=spatial,
        sun_xy_kpc=sun_xy,
        radius_kpc=inputs.radius_kpc,
    )

    keys = ["alive", "rsg", "cburn", "cburn_rsg", "within", "cburn_rsg_within"]
    res = np.zeros((inputs.n_sims, len(keys)), dtype=float)
    for i in range(inputs.n_sims):
        d = _simulate_once(
            rng=rng,
            bins=bins,
            imf_name=inputs.imf,
            sfr_msun_per_yr=inputs.sfr_msun_per_yr,
            duration_myr=inputs.duration_myr,
            p_within=p_within,
            fast=inputs.fast,
        )
        res[i] = [d[k] for k in keys]

    mean = {k: float(res[:, j].mean()) for j, k in enumerate(keys)}
    std = {k: float(res[:, j].std()) for j, k in enumerate(keys)}
    summary = PopulationSummary(p_within=float(p_within), mean=mean, std=std)

    print(
        f"IMF={inputs.imf}, SFR={inputs.sfr_msun_per_yr} Msun/yr, duration={inputs.duration_myr} Myr, "
        f"sims={inputs.n_sims}, P(within {inputs.radius_kpc} kpc)≈{p_within:.6g}"
    )
    for k in keys:
        print(f"{k}: {mean[k]:.3f} ± {std[k]:.3f}")
    if mean["rsg"] > 0:
        print(f"cburn_rsg / rsg: {(mean['cburn_rsg'] / mean['rsg']):.4f}")
    if mean["alive"] > 0:
        print(f"rsg / alive: {(mean['rsg'] / mean['alive']):.4f}")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "inputs": {
                        **inputs.__dict__,
                        "phases_csv": str(inputs.phases_csv),
                    },
                    "p_within": summary.p_within,
                    "mean": summary.mean,
                    "std": summary.std,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if out_plot is not None:
        import os

        tmp = Path(os.environ.get("TMPDIR", "/tmp"))
        Path(os.environ.setdefault("MPLCONFIGDIR", str(tmp / "matplotlib"))).mkdir(parents=True, exist_ok=True)
        Path(os.environ.setdefault("XDG_CACHE_HOME", str(tmp / "xdg_cache"))).mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        out_plot.parent.mkdir(parents=True, exist_ok=True)
        y = [mean[k] for k in keys]
        yerr = [std[k] for k in keys]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(keys)), y, yerr=yerr, capsize=3)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"MW toy population (IMF={inputs.imf}, radius={inputs.radius_kpc} kpc)")
        fig.tight_layout()
        fig.savefig(out_plot, dpi=200)
        plt.close(fig)

    return summary


def simulate_snapshot_catalog(
    *,
    phases_csv: Path | str,
    imf: str,
    sfr_msun_per_yr: float,
    t_obs_myr: float,
    seed: int = 1,
    spatial_params: MWYoungSFParams | None = None,
) -> SnapshotCatalog:
    """
    Draw a single Monte Carlo *catalog* at observation time t_obs_myr.

    Model:
    - constant SFR from t=0 to t=t_obs_myr
    - births in each mass bin set by the IMF and SFR
    - each star's age is uniform in [0, t_obs_myr]
    - a star is alive if age < lifetime
    - RSG and C-burning are determined from the phase windows in the CSV

    This is mainly for visualization (e.g. 2D MW map with colors).
    """
    if t_obs_myr <= 0:
        raise ValueError("t_obs_myr must be positive")

    points = load_phases_csv(phases_csv)
    bins = make_mass_bins(points)
    imf_obj = imf_preset(imf)
    rng = np.random.default_rng(seed)
    spatial = MWYoungSFParams() if spatial_params is None else spatial_params

    masses: list[float] = []
    ages: list[float] = []
    alive: list[bool] = []
    is_rsg: list[bool] = []
    is_cburn: list[bool] = []

    for b in bins:
        births_per_myr = sfr_msun_per_yr * 1e6 * imf_obj.number_per_msun(b.m_lo, b.m_hi)
        mean_births = births_per_myr * t_obs_myr
        n = int(rng.poisson(mean_births)) if mean_births > 0 else 0
        if n <= 0:
            continue

        a = rng.uniform(0.0, t_obs_myr, size=n)
        lif = float(b.lifetime_myr)
        alive_mask = a < lif

        rsg_start = max(0.0, lif - float(b.rsg_duration_myr))
        rsg_end = lif

        if b.cburn_start_myr is not None and b.cburn_end_myr is not None:
            cb0, cb1 = float(b.cburn_start_myr), float(b.cburn_end_myr)
        else:
            cb1 = lif
            cb0 = max(0.0, cb1 - float(b.cburn_duration_myr))

        rsg_mask = alive_mask & (a >= rsg_start) & (a <= rsg_end)
        cburn_mask = alive_mask & (a >= cb0) & (a <= cb1)

        masses.append(np.full(n, float(b.m_center), dtype=float))
        ages.append(a.astype(float))
        alive.append(alive_mask.astype(bool))
        is_rsg.append(rsg_mask.astype(bool))
        is_cburn.append(cburn_mask.astype(bool))

    if not masses:
        z = np.zeros(0, dtype=float)
        b = np.zeros(0, dtype=bool)
        return SnapshotCatalog(z, z, z, z, b, b, b)

    mass_arr = np.concatenate(masses)
    age_arr = np.concatenate(ages)
    alive_arr = np.concatenate(alive)
    rsg_arr = np.concatenate(is_rsg)
    cburn_arr = np.concatenate(is_cburn)

    x, y = sample_mw_young_xy(mass_arr.size, rng, spatial)
    return SnapshotCatalog(
        x_kpc=x,
        y_kpc=y,
        mass_msun=mass_arr,
        age_myr=age_arr,
        alive=alive_arr,
        is_rsg=rsg_arr,
        is_cburn=cburn_arr,
    )


def run_imf_scan(
    *,
    phases_csv: Path,
    imfs: list[str],
    sfr_msun_per_yr: float = 2.0,
    duration_myr: float = 20.0,
    n_sims: int = 10_000,
    seed: int = 1,
    radius_kpc: float = 1.0,
    sun_x_kpc: float = -8.2,
    sun_y_kpc: float = 0.0,
    within_samples: int = 200_000,
    fast: bool = True,
    out_csv: Path | None = None,
    out_plot: Path | None = None,
) -> list[IMFScanRow]:
    """
    Run the same toy population Monte Carlo for multiple IMFs and summarize the
    expected number of C-burning RSGs (and related quantities).
    """
    if not imfs:
        raise ValueError("imfs must be non-empty")

    points = load_phases_csv(phases_csv)
    bins = _make_mass_bins(points)

    rng_shared = np.random.default_rng(seed)
    spatial = MWYoungSFParams()
    sun_xy = (sun_x_kpc, sun_y_kpc)
    p_within = estimate_within_probability(
        rng=rng_shared,
        n_samples=within_samples,
        spatial_params=spatial,
        sun_xy_kpc=sun_xy,
        radius_kpc=radius_kpc,
    )

    keys = ["alive", "rsg", "cburn", "cburn_rsg", "within", "cburn_rsg_within"]
    rows: list[IMFScanRow] = []
    for imf in imfs:
        rng = np.random.default_rng(seed + (abs(hash(imf)) % 1_000_000))
        res = np.zeros((n_sims, len(keys)), dtype=float)
        for i in range(n_sims):
            d = _simulate_once(
                rng=rng,
                bins=bins,
                imf_name=imf,
                sfr_msun_per_yr=sfr_msun_per_yr,
                duration_myr=duration_myr,
                p_within=p_within,
                fast=fast,
            )
            res[i] = [d[k] for k in keys]

        mean = {k: float(res[:, j].mean()) for j, k in enumerate(keys)}
        std = {k: float(res[:, j].std()) for j, k in enumerate(keys)}
        ratios: dict[str, float] = {}
        ratios["cburn_rsg_over_rsg"] = (mean["cburn_rsg"] / mean["rsg"]) if mean["rsg"] > 0 else float("nan")
        ratios["rsg_over_alive"] = (mean["rsg"] / mean["alive"]) if mean["alive"] > 0 else float("nan")
        rows.append(IMFScanRow(imf=imf, p_within=float(p_within), mean=mean, std=std, ratios=ratios))

    print(f"P(within {radius_kpc} kpc)≈{p_within:.6g} (shared)")
    header = (
        "IMF".ljust(10)
        + "cburn_rsg(MW)".rjust(16)
        + "std".rjust(10)
        + f"cburn_rsg(<{radius_kpc:g}kpc)".rjust(22)
        + "std".rjust(10)
        + "ratio".rjust(10)
    )
    print(header)
    for r in rows:
        print(
            r.imf.ljust(10)
            + f"{r.mean['cburn_rsg']:16.3f}"
            + f"{r.std['cburn_rsg']:10.3f}"
            + f"{r.mean['cburn_rsg_within']:22.4f}"
            + f"{r.std['cburn_rsg_within']:10.4f}"
            + f"{r.ratios['cburn_rsg_over_rsg']:10.4f}"
        )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "imf",
                    "p_within",
                    "alive_mean",
                    "alive_std",
                    "rsg_mean",
                    "rsg_std",
                    "cburn_mean",
                    "cburn_std",
                    "cburn_rsg_mean",
                    "cburn_rsg_std",
                    "cburn_rsg_within_mean",
                    "cburn_rsg_within_std",
                    "cburn_rsg_over_rsg",
                    "rsg_over_alive",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "imf": r.imf,
                        "p_within": f"{r.p_within:.6g}",
                        "alive_mean": f"{r.mean['alive']:.6f}",
                        "alive_std": f"{r.std['alive']:.6f}",
                        "rsg_mean": f"{r.mean['rsg']:.6f}",
                        "rsg_std": f"{r.std['rsg']:.6f}",
                        "cburn_mean": f"{r.mean['cburn']:.6f}",
                        "cburn_std": f"{r.std['cburn']:.6f}",
                        "cburn_rsg_mean": f"{r.mean['cburn_rsg']:.6f}",
                        "cburn_rsg_std": f"{r.std['cburn_rsg']:.6f}",
                        "cburn_rsg_within_mean": f"{r.mean['cburn_rsg_within']:.6f}",
                        "cburn_rsg_within_std": f"{r.std['cburn_rsg_within']:.6f}",
                        "cburn_rsg_over_rsg": f"{r.ratios['cburn_rsg_over_rsg']:.6g}",
                        "rsg_over_alive": f"{r.ratios['rsg_over_alive']:.6g}",
                    }
                )

    if out_plot is not None:
        import os

        tmp = Path(os.environ.get("TMPDIR", "/tmp"))
        Path(os.environ.setdefault("MPLCONFIGDIR", str(tmp / "matplotlib"))).mkdir(parents=True, exist_ok=True)
        Path(os.environ.setdefault("XDG_CACHE_HOME", str(tmp / "xdg_cache"))).mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        out_plot.parent.mkdir(parents=True, exist_ok=True)
        xs = np.arange(len(rows))
        y = [r.mean["cburn_rsg"] for r in rows]
        yerr = [r.std["cburn_rsg"] for r in rows]
        labels = [r.imf for r in rows]

        y_local = [r.mean["cburn_rsg_within"] for r in rows]
        yerr_local = [r.std["cburn_rsg_within"] for r in rows]

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)
        ax0.bar(xs, y, yerr=yerr, capsize=3)
        ax0.set_ylabel("Expected count (whole MW)")
        ax0.set_title("C-burning RSGs vs IMF (toy model)")

        ax1.bar(xs, y_local, yerr=yerr_local, capsize=3, color="#9467bd")
        ax1.set_ylabel(f"Expected count (<{radius_kpc:g} kpc)")
        ax1.set_xticks(xs)
        ax1.set_xticklabels(labels)
        fig.tight_layout()
        fig.savefig(out_plot, dpi=200)
        plt.close(fig)

    return rows
