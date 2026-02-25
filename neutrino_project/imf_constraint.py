from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detectability import DetectorPreset, detector_presets, estimate_detectability_at_time


@dataclass(frozen=True)
class IMFConstraintResult:
    """
    Simple IMF-discrimination summary for a signal-only Poisson counting experiment.

    This is a pedagogical tool:
    - it ignores backgrounds unless you provide a background rate
    - it assumes rates scale linearly with detector fiducial mass (true for ES target count)
    - it does not include systematics (SFR uncertainty, model uncertainty in Lnu, etc.)
    """

    imf_a: str
    imf_b: str
    detector_name: str
    exposure_years: float
    z_sigma: float
    base_mass_kton: float
    base_rate_a_per_year: float
    base_rate_b_per_year: float
    base_background_per_year: float
    required_mass_kton_for_z: float


def required_scale_for_z(
    *,
    rate_a_per_year: float,
    rate_b_per_year: float,
    background_per_year: float,
    exposure_years: float,
    z_sigma: float,
) -> float:
    """
    Required multiplicative scale 's' on (signal + background) to achieve a z-sigma
    separation using a simple Gaussianized Poisson approximation:

      |(s*ra - s*rb)*T| >= z * sqrt((s*ra + s*bkg)*T)

    Solve for s:
      s >= z^2 * (ra + bkg) / ((ra - rb)^2 * T)
    """
    ra = float(rate_a_per_year)
    rb = float(rate_b_per_year)
    bkg = float(background_per_year)
    T = float(exposure_years)
    z = float(z_sigma)

    if T <= 0:
        raise ValueError("exposure_years must be positive")
    if z <= 0:
        raise ValueError("z_sigma must be positive")
    if ra < 0 or rb < 0 or bkg < 0:
        raise ValueError("rates must be non-negative")

    dr = abs(ra - rb)
    if dr <= 0:
        return float("inf")

    return (z * z) * (ra + bkg) / ((dr * dr) * T)


def imf_constraint_matrix(
    *,
    phases_csv: Path | str,
    imfs: list[str],
    detector: DetectorPreset,
    exposure_years: float,
    z_sigma: float = 3.0,
    background_per_year: float = 0.0,
    # model parameters
    sfr_msun_per_yr: float = 2.0,
    t_obs_myr: float = 20.0,
    radius_kpc: float = 1.0,
    sun_xy_kpc: tuple[float, float] = (-8.2, 0.0),
    within_samples: int = 200_000,
    inv_d2_samples: int = 100_000,
    seed: int = 1,
    lnu_per_star_erg_s: float = 1e38,
    mean_energy_mev: float = 0.9,
    alpha: float = 2.0,
) -> tuple[list[IMFConstraintResult], np.ndarray]:
    """
    Compare all pairs of IMFs and compute the required detector mass to separate them at z_sigma.

    Returns:
    - flat list of results for (a,b) pairs
    - a matrix M where M[i,j] is required mass in kton to separate imfs[i] vs imfs[j]
    """
    if not imfs:
        raise ValueError("imfs must be non-empty")

    # Compute base rates at the given detector mass for each IMF.
    rates = {}
    for imf in imfs:
        d = estimate_detectability_at_time(
            phases_csv=phases_csv,
            imf=imf,
            sfr_msun_per_yr=sfr_msun_per_yr,
            t_obs_myr=t_obs_myr,
            radius_kpc=radius_kpc,
            sun_xy_kpc=sun_xy_kpc,
            within_samples=within_samples,
            inv_d2_samples=inv_d2_samples,
            seed=seed,
            lnu_per_star_erg_s=lnu_per_star_erg_s,
            mean_energy_mev=mean_energy_mev,
            alpha=alpha,
            detector_list=[detector],
        )
        rates[imf] = d.events_per_year_by_detector[detector.name]

    base_mass = float(detector.fiducial_mass_kton)
    results: list[IMFConstraintResult] = []
    mat = np.full((len(imfs), len(imfs)), np.nan, dtype=float)
    for i, a in enumerate(imfs):
        for j, b in enumerate(imfs):
            ra = float(rates[a])
            rb = float(rates[b])
            scale = required_scale_for_z(
                rate_a_per_year=ra,
                rate_b_per_year=rb,
                background_per_year=background_per_year,
                exposure_years=exposure_years,
                z_sigma=z_sigma,
            )
            req_mass = float("inf") if not np.isfinite(scale) else base_mass * max(1.0, scale)
            mat[i, j] = req_mass
            if i < j:
                results.append(
                    IMFConstraintResult(
                        imf_a=a,
                        imf_b=b,
                        detector_name=detector.name,
                        exposure_years=float(exposure_years),
                        z_sigma=float(z_sigma),
                        base_mass_kton=base_mass,
                        base_rate_a_per_year=ra,
                        base_rate_b_per_year=rb,
                        base_background_per_year=float(background_per_year),
                        required_mass_kton_for_z=req_mass,
                    )
                )
    return results, mat


def default_future_detector() -> DetectorPreset:
    # A deliberately "hopeful" concept-scale benchmark.
    return DetectorPreset(name="THEIA-like (100 kt, concept)", fiducial_mass_kton=100.0)

