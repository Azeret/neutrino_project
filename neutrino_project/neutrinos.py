from __future__ import annotations

"""
Toy neutrino utilities.

This module is intentionally *not* a detailed reproduction of neutrinos.pdf
(Seong et al. 2025), which uses MESA stellar models + detailed neutrino
emissivities and detector simulations.

How to proceed toward reproducing the paper more faithfully
----------------------------------------------------------
1) Use MESA (or published MESA outputs) to compute L_ν(t), flavor-resolved, for
   the relevant evolutionary phases (RSG/C-burning focus).
2) Use thermal-process emissivities (pair/photo/brem/recomb/plasmon) consistent
   with the stellar structure, and a physically motivated spectrum per process.
3) Include neutrino oscillations between source and detector (and Earth matter
   effects if needed).
4) Replace the toy ES σ(E) with a standard calculation and add detector
   response + thresholds + backgrounds.
"""

from dataclasses import dataclass

import numpy as np


_NA = 6.02214076e23  # 1/mol
_MEV_TO_ERG = 1.602176634e-6
_SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class Detector:
    name: str
    n_electrons: float

    @staticmethod
    def water_equivalent(*, fiducial_mass_kton: float, name: str = "water") -> "Detector":
        """
        Electron targets for neutrino-electron elastic scattering in water.

        Water (H2O):
        - molar mass = 18 g/mol
        - electrons per molecule = 10
        """
        if fiducial_mass_kton <= 0:
            raise ValueError("fiducial_mass_kton must be positive")
        mass_g = fiducial_mass_kton * 1e9
        n_e = mass_g * (10.0 * _NA / 18.0)
        return Detector(name=name, n_electrons=float(n_e))


@dataclass(frozen=True)
class NeutrinoSpectrumModel:
    """
    A toy spectral shape for *number* spectrum.

    The alpha-fit is a gamma distribution with mean <E> and pinching alpha.
    """

    mean_energy_mev: float
    alpha: float

    @staticmethod
    def alpha_fit(*, mean_energy_mev: float, alpha: float = 2.0) -> "NeutrinoSpectrumModel":
        if mean_energy_mev <= 0:
            raise ValueError("mean_energy_mev must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        return NeutrinoSpectrumModel(mean_energy_mev=float(mean_energy_mev), alpha=float(alpha))

    def pdf(self, e_mev: np.ndarray) -> np.ndarray:
        """
        Probability density f(E) with ∫ f(E) dE = 1 over [0, ∞).
        """
        e = np.asarray(e_mev, dtype=float)
        if np.any(e < 0):
            raise ValueError("Energy must be non-negative")
        k = self.alpha + 1.0
        theta = self.mean_energy_mev / (self.alpha + 1.0)
        from math import gamma

        norm = 1.0 / (gamma(k) * (theta**k))
        return norm * (e ** (k - 1.0)) * np.exp(-e / theta)


def flux_from_luminosity(
    *,
    lnu_erg_s: float,
    mean_energy_mev: float,
    distance_pc: float,
) -> float:
    """
    Convert energy luminosity L_nu to total number flux at Earth.

    Φ_total = (L_nu / <E>) / (4π d^2)
    """
    if lnu_erg_s <= 0:
        raise ValueError("lnu_erg_s must be positive")
    if mean_energy_mev <= 0:
        raise ValueError("mean_energy_mev must be positive")
    if distance_pc <= 0:
        raise ValueError("distance_pc must be positive")
    e_erg = mean_energy_mev * _MEV_TO_ERG
    n_dot = lnu_erg_s / e_erg
    d_cm = distance_pc * 3.085677581e18
    return float(n_dot / (4.0 * np.pi * d_cm * d_cm))


def _sigma_es_cm2(e_mev: np.ndarray, flavor: str) -> np.ndarray:
    """
    Toy total cross sections for ν-e elastic scattering (order-of-magnitude).

    A common low-energy approximation is σ ∝ Eν.
    """
    e = np.asarray(e_mev, dtype=float)
    if np.any(e < 0):
        raise ValueError("Energy must be non-negative")
    flavor = flavor.strip().lower()
    if flavor == "nue":
        c = 9.20e-45
    elif flavor in {"numu", "nutau"}:
        c = 1.57e-45
    else:
        raise ValueError("flavor must be one of: nue, numu, nutau")
    return c * e


def estimate_es_events_per_year(
    *,
    flux_at_earth_cm2_s: float,
    detector: Detector,
    spectrum: NeutrinoSpectrumModel,
    e_min_mev: float = 0.1,
    e_max_mev: float = 5.0,
    n_grid: int = 2000,
    flavor: str = "nue",
) -> float:
    """
    Estimate elastic-scattering (ES) events/year: ν + e -> ν + e.

    Uses:
      rate = N_e ∫ dE [ Φ_total f(E) ] σ(E)
    """
    if flux_at_earth_cm2_s < 0:
        raise ValueError("flux_at_earth_cm2_s must be non-negative")
    if e_max_mev <= e_min_mev:
        raise ValueError("e_max_mev must be > e_min_mev")
    if n_grid < 10:
        raise ValueError("n_grid too small")

    e = np.linspace(e_min_mev, e_max_mev, int(n_grid))
    f = spectrum.pdf(e)
    f /= np.trapz(f, e)

    phi_e = flux_at_earth_cm2_s * f
    sigma = _sigma_es_cm2(e, flavor=flavor)
    rate_s = detector.n_electrons * float(np.trapz(phi_e * sigma, e))
    return rate_s * _SECONDS_PER_YEAR

