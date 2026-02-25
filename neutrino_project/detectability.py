from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .galaxy import MWYoungSFParams, estimate_within_probability, sample_mw_young_xy
from .neutrinos import Detector, NeutrinoSpectrumModel, estimate_es_events_per_year
from .population import expected_counts_vs_time


_KPC_TO_CM = 3.085677581e21
_MEV_TO_ERG = 1.602176634e-6


@dataclass(frozen=True)
class DetectorPreset:
    name: str
    fiducial_mass_kton: float

    def to_detector(self) -> Detector:
        return Detector.water_equivalent(fiducial_mass_kton=self.fiducial_mass_kton, name=self.name)


def detector_presets() -> list[DetectorPreset]:
    """
    A few water-equivalent fiducial masses for quick comparisons.

    Notes:
    - This is *not* detector modeling. It's just target scaling for ES.
    - Backgrounds, thresholds, efficiencies are not included.
    """
    return [
        DetectorPreset(name="SK-like (22.5 kt)", fiducial_mass_kton=22.5),
        DetectorPreset(name="Hyper-K-like (187 kt)", fiducial_mass_kton=187.0),
        DetectorPreset(name="1 Mt water (toy)", fiducial_mass_kton=1000.0),
    ]


def mean_inv_d2_within_radius(
    *,
    n_samples: int,
    radius_kpc: float,
    sun_xy_kpc: tuple[float, float],
    seed: int,
    spatial_params: MWYoungSFParams | None = None,
    d_floor_kpc: float = 0.05,
) -> float:
    """
    Return E[1/d^2 | d <= radius] under the toy spatial model.

    A small distance floor avoids the 1/d^2 divergence as d->0 in Monte Carlo.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if radius_kpc <= 0:
        raise ValueError("radius_kpc must be positive")

    rng = np.random.default_rng(seed)
    p = MWYoungSFParams() if spatial_params is None else spatial_params
    x, y = sample_mw_young_xy(n_samples, rng, p)
    sx, sy = sun_xy_kpc
    d_kpc = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
    sel = d_kpc <= radius_kpc
    if not np.any(sel):
        return float("nan")
    d_cm = np.clip(d_kpc[sel] * _KPC_TO_CM, d_floor_kpc * _KPC_TO_CM, None)
    return float(np.mean(1.0 / (d_cm * d_cm)))


def one_star_flux_cm2_s(
    *,
    lnu_erg_s: float,
    mean_energy_mev: float,
    inv_d2_cm2: float,
) -> float:
    """
    Expected number flux contribution of ONE star when averaging over 1/d^2:

      Φ1 = (Lν / <E>) * (1 / (4π)) * <1/d^2>
    """
    if lnu_erg_s <= 0:
        raise ValueError("lnu_erg_s must be positive")
    if mean_energy_mev <= 0:
        raise ValueError("mean_energy_mev must be positive")
    if inv_d2_cm2 <= 0:
        raise ValueError("inv_d2_cm2 must be positive")
    e_erg = mean_energy_mev * _MEV_TO_ERG
    n_dot = lnu_erg_s / e_erg
    return float(n_dot * (inv_d2_cm2 / (4.0 * np.pi)))


@dataclass(frozen=True)
class DetectabilityAtTime:
    t_myr: float
    p_within: float
    n_cburn_rsg_mw: float
    n_cburn_rsg_within: float
    flux_within_cm2_s: float
    events_per_year_by_detector: dict[str, float]


def estimate_detectability_at_time(
    *,
    phases_csv: Path | str,
    imf: str,
    sfr_msun_per_yr: float,
    t_obs_myr: float,
    radius_kpc: float,
    sun_xy_kpc: tuple[float, float] = (-8.2, 0.0),
    within_samples: int = 200_000,
    inv_d2_samples: int = 100_000,
    seed: int = 1,
    lnu_per_star_erg_s: float = 1e38,
    mean_energy_mev: float = 0.9,
    alpha: float = 2.0,
    detector_list: list[DetectorPreset] | None = None,
    flavor: str = "nue",
) -> DetectabilityAtTime:
    """
    Pipeline:
    1) compute expected C-burning RSG counts in the MW at t_obs
    2) compute expected local counts within radius around the Sun using p_within
    3) convert local counts -> flux at Earth using <1/d^2 | within>
    4) convert flux -> toy ES events/year for a few detector masses
    """
    if t_obs_myr <= 0:
        raise ValueError("t_obs_myr must be positive")

    spatial = MWYoungSFParams()
    rng = np.random.default_rng(seed)
    p_within = estimate_within_probability(
        rng=rng,
        n_samples=within_samples,
        spatial_params=spatial,
        sun_xy_kpc=sun_xy_kpc,
        radius_kpc=radius_kpc,
    )
    inv_d2 = mean_inv_d2_within_radius(
        n_samples=inv_d2_samples,
        radius_kpc=radius_kpc,
        sun_xy_kpc=sun_xy_kpc,
        seed=seed + 999,
        spatial_params=spatial,
    )

    d = expected_counts_vs_time(
        phases_csv=phases_csv,
        imf=imf,
        sfr_msun_per_yr=sfr_msun_per_yr,
        t_grid_myr=np.array([t_obs_myr], dtype=float),
        radius_kpc=radius_kpc,
        sun_x_kpc=sun_xy_kpc[0],
        sun_y_kpc=sun_xy_kpc[1],
        within_samples=within_samples,
        seed=seed,
        p_within_override=p_within,
    )
    n_mw = float(d["cburn_rsg"][-1])
    n_within = float(d["cburn_rsg_within"][-1])

    phi1 = one_star_flux_cm2_s(lnu_erg_s=lnu_per_star_erg_s, mean_energy_mev=mean_energy_mev, inv_d2_cm2=inv_d2)
    flux = n_within * phi1

    spec = NeutrinoSpectrumModel.alpha_fit(mean_energy_mev=mean_energy_mev, alpha=alpha)
    dets = detector_presets() if detector_list is None else detector_list
    events = {}
    for preset in dets:
        det = preset.to_detector()
        events[det.name] = float(
            estimate_es_events_per_year(
                flux_at_earth_cm2_s=flux,
                detector=det,
                spectrum=spec,
                flavor=flavor,
            )
        )

    return DetectabilityAtTime(
        t_myr=float(t_obs_myr),
        p_within=float(p_within),
        n_cburn_rsg_mw=n_mw,
        n_cburn_rsg_within=n_within,
        flux_within_cm2_s=float(flux),
        events_per_year_by_detector=events,
    )
