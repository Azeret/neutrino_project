from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detectability import DetectorPreset, estimate_detectability_at_time


@dataclass(frozen=True)
class IMFConstraintResult:
    """
    Simple IMF-discrimination summary for a toy Poisson counting experiment.

    This is a pedagogical tool:
    - it can include a simple background model (background ∝ detector mass)
    - it assumes rates scale linearly with detector fiducial mass (true for ES target count)
    - it does not include systematics (SFR uncertainty, model uncertainty in Lnu, etc.)
    """

    imf_a: str
    imf_b: str
    method: str
    detector_name: str
    exposure_years: float
    z_sigma: float
    base_mass_kton: float
    base_rate_a_per_year: float
    base_rate_b_per_year: float
    include_background: bool
    background_per_kton_year: float
    base_background_per_year: float
    required_mass_kton_for_z: float


def _kl_divergence_poisson(lam_true: float, lam_alt: float) -> float:
    """
    KL divergence D(P_true || P_alt) for Poisson distributions with means lam_true and lam_alt:
      D = lam_true * ln(lam_true/lam_alt) + lam_alt - lam_true
    """
    lt = float(lam_true)
    la = float(lam_alt)
    if lt < 0 or la < 0:
        raise ValueError("Poisson means must be non-negative")
    if lt == 0 and la == 0:
        return 0.0
    if la == 0 and lt > 0:
        return float("inf")
    if lt == 0:
        # D(0 || la) = la
        return la
    return lt * np.log(lt / la) + la - lt


def required_scale_for_z_asimov_symmetric(
    *,
    rate_a_per_year: float,
    rate_b_per_year: float,
    background_per_year: float,
    exposure_years: float,
    z_sigma: float,
) -> float:
    """
    Required multiplicative scale 's' on (signal + background) to achieve a z-sigma separation,
    using an Asimov (likelihood-ratio) approximation for two Poisson hypotheses.

    Let the *total* mean rates (signal+background) be:
      R_a = ra + bkg
      R_b = rb + bkg
    Then the Poisson means over exposure T are:
      λ_a = s * R_a * T
      λ_b = s * R_b * T

    The Asimov separation for "a vs b" is:
      Z(a||b) ≈ sqrt(2 * D(λ_a || λ_b))
    where D is the Poisson KL divergence.

    This function is symmetric: it returns the scale needed so that
      min(Z(a||b), Z(b||a)) >= z_sigma.

    Note: because D(sx||sy) = s D(x||y), the required scale is:
      s >= z^2 / (2 * D0)
    where D0 is computed at s=1.
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

    lam_a = (ra + bkg) * T
    lam_b = (rb + bkg) * T
    if lam_a == lam_b:
        return float("inf")

    d_ab = _kl_divergence_poisson(lam_a, lam_b)
    d_ba = _kl_divergence_poisson(lam_b, lam_a)
    d0 = min(d_ab, d_ba)
    if d0 <= 0:
        return float("inf")

    return (z * z) / (2.0 * d0)


def required_scale_for_z_gaussian_symmetric(
    *,
    rate_a_per_year: float,
    rate_b_per_year: float,
    background_per_year: float,
    exposure_years: float,
    z_sigma: float,
) -> float:
    """
    Required scale using a simple Gaussianized Poisson approximation.

    For each "true" hypothesis (a or b), require:
      |μ_a - μ_b| >= z * sqrt(μ_true)
    where μ_true = (s*(r_true + bkg))*T and μ's include background.

    Returns the larger (more conservative) scale needed across both truth assignments.
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

    # s >= z^2 * (r_true + bkg) / ((ra-rb)^2 * T)
    s_a = (z * z) * (ra + bkg) / ((dr * dr) * T)
    s_b = (z * z) * (rb + bkg) / ((dr * dr) * T)
    return max(s_a, s_b)


def _method_to_scale_fn(method: str):
    m = method.strip().lower()
    if m in ("asimov", "llr", "likelihood"):
        return required_scale_for_z_asimov_symmetric, "asimov"
    if m in ("gaussian", "gauss", "normal"):
        return required_scale_for_z_gaussian_symmetric, "gaussian"
    raise ValueError(f"Unknown method={method!r}. Use 'asimov' or 'gaussian'.")


def imf_constraint_matrix(
    *,
    phases_csv: Path | str,
    imfs: list[str],
    detector: DetectorPreset,
    exposure_years: float,
    z_sigma: float = 3.0,
    include_background: bool = True,
    background_per_kton_year: float = 0.0,
    method: str = "asimov",
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

    scale_fn, method_name = _method_to_scale_fn(method)

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
    bkg_per_year_base = float(background_per_kton_year) * base_mass if include_background else 0.0
    results: list[IMFConstraintResult] = []
    mat = np.full((len(imfs), len(imfs)), np.nan, dtype=float)
    for i, a in enumerate(imfs):
        for j, b in enumerate(imfs):
            ra = float(rates[a])
            rb = float(rates[b])
            scale = scale_fn(
                rate_a_per_year=ra,
                rate_b_per_year=rb,
                background_per_year=bkg_per_year_base,
                exposure_years=exposure_years,
                z_sigma=z_sigma,
            )
            req_mass = float("inf") if not np.isfinite(scale) else base_mass * max(0.0, scale)
            mat[i, j] = req_mass
            if i < j:
                results.append(
                    IMFConstraintResult(
                        imf_a=a,
                        imf_b=b,
                        method=method_name,
                        detector_name=detector.name,
                        exposure_years=float(exposure_years),
                        z_sigma=float(z_sigma),
                        base_mass_kton=base_mass,
                        base_rate_a_per_year=ra,
                        base_rate_b_per_year=rb,
                        include_background=bool(include_background),
                        background_per_kton_year=float(background_per_kton_year),
                        base_background_per_year=float(bkg_per_year_base),
                        required_mass_kton_for_z=req_mass,
                    )
                )
    return results, mat


def default_future_detector() -> DetectorPreset:
    # A deliberately "hopeful" concept-scale benchmark.
    return DetectorPreset(name="THEIA-like (100 kt, concept)", fiducial_mass_kton=100.0)
