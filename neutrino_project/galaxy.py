from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MWYoungSFParams:
    """
    Toy spatial model for *young* massive stars in the Milky Way disk (kpc).

    Components:
    - exponential disk surface density Σ(R) ∝ exp(-R/R_scale)
    - logarithmic spiral arms
    - a molecular ring

    This is intentionally simple and is meant for order-of-magnitude estimates.
    """

    R_scale: float = 2.6
    R_min: float = 0.5
    R_max: float = 15.0
    m_arms: int = 4
    pitch_deg: float = 12.0
    sigma_arm: float = 0.30
    f_arm: float = 0.65
    R_ref: float = 5.0
    phi0_deg: float = 20.0
    ring_R0: float = 4.5
    ring_sigma: float = 0.5
    f_ring: float = 0.10


def sample_exponential_R(
    n: int,
    R_scale: float,
    R_min: float,
    R_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw R from a truncated exponential Σ(R) ∝ exp(-R/R_scale),
    including the area element 2πR dR via rejection sampling.
    """
    R = np.empty(n, dtype=float)
    i = 0
    env_norm = 1.0 - np.exp(-(R_max - R_min) / R_scale)
    while i < n:
        u1, u2 = rng.random(2)
        R_prop = R_min - R_scale * np.log(1.0 - u1 * env_norm)
        acc = R_prop / R_max
        if (R_prop <= R_max) and (u2 < acc):
            R[i] = R_prop
            i += 1
    return R


def log_spiral_R(phi: np.ndarray, R_ref: float, pitch_rad: float, phi0: np.ndarray) -> np.ndarray:
    return R_ref * np.exp((phi - phi0) * np.tan(pitch_rad))


def sample_mw_young_xy(
    n: int,
    rng: np.random.Generator,
    p: MWYoungSFParams,
) -> tuple[np.ndarray, np.ndarray]:
    pitch = np.deg2rad(p.pitch_deg)
    phi0 = np.deg2rad(p.phi0_deg)

    n_arm = int(np.round(float(p.f_arm) * n))
    n_ring = int(np.round(float(p.f_ring) * n))
    n_disk = n - n_arm - n_ring
    if n_disk < 0:
        n_disk = 0
        n_arm = min(n, n_arm)
        n_ring = max(0, n - n_arm)

    x = np.empty(n, dtype=float)
    y = np.empty(n, dtype=float)
    idx = 0

    if n_disk > 0:
        R_disk = sample_exponential_R(n_disk, p.R_scale, p.R_min, p.R_max, rng)
        phi_disk = rng.uniform(0.0, 2.0 * np.pi, n_disk)
        x[idx : idx + n_disk] = R_disk * np.cos(phi_disk)
        y[idx : idx + n_disk] = R_disk * np.sin(phi_disk)
        idx += n_disk

    if n_ring > 0:
        R_ring = rng.normal(p.ring_R0, p.ring_sigma, n_ring)
        R_ring = np.clip(R_ring, p.R_min, p.R_max)
        phi_ring = rng.uniform(0.0, 2.0 * np.pi, n_ring)
        x[idx : idx + n_ring] = R_ring * np.cos(phi_ring)
        y[idx : idx + n_ring] = R_ring * np.sin(phi_ring)
        idx += n_ring

    if n_arm > 0:
        arm_idx = rng.integers(0, p.m_arms, n_arm)
        n_turns = 1.5
        dphi_arm = n_turns * 2.0 * np.pi / p.m_arms
        R_min_arm, R_max_arm = 3.0, 13.0

        phi_arm0 = phi0 + 2.0 * np.pi * arm_idx / p.m_arms
        phi_arm = phi_arm0 + rng.uniform(0.0, dphi_arm, n_arm)
        R_center = log_spiral_R(phi_arm, p.R_ref, pitch, phi_arm0)
        R_arm = R_center + rng.normal(0.0, p.sigma_arm, n_arm)

        bad = (R_arm <= R_min_arm) | (R_arm >= R_max_arm)
        if bad.any():
            R_arm[bad] = sample_exponential_R(int(bad.sum()), p.R_scale, R_min_arm, R_max_arm, rng)

        x[idx : idx + n_arm] = R_arm * np.cos(phi_arm)
        y[idx : idx + n_arm] = R_arm * np.sin(phi_arm)

    perm = rng.permutation(n)
    return x[perm], y[perm]


def estimate_within_probability(
    *,
    rng: np.random.Generator,
    n_samples: int,
    spatial_params: MWYoungSFParams,
    sun_xy_kpc: tuple[float, float],
    radius_kpc: float,
) -> float:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    x, y = sample_mw_young_xy(n_samples, rng, spatial_params)
    sx, sy = sun_xy_kpc
    dist = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
    return float((dist <= radius_kpc).mean())

