from __future__ import annotations

from dataclasses import dataclass


def _powerlaw_int(m0: float, m1: float, alpha: float) -> float:
    # ∫ m^{-alpha} dm from m0 to m1
    if m1 <= m0:
        return 0.0
    if abs(alpha - 1.0) < 1e-12:
        return float(__import__("math").log(m1 / m0))
    return (m1 ** (1.0 - alpha) - m0 ** (1.0 - alpha)) / (1.0 - alpha)


def _powerlaw_mass_int(m0: float, m1: float, alpha: float) -> float:
    # ∫ m * m^{-alpha} dm = ∫ m^{1-alpha} dm
    if m1 <= m0:
        return 0.0
    p = 2.0 - alpha
    if abs(p) < 1e-12:
        return float(__import__("math").log(m1 / m0))
    return (m1**p - m0**p) / p


@dataclass(frozen=True)
class TwoPartIMF:
    """
    Two-part broken power-law IMF:

    ξ(m) = k1 m^{-α1} for m in [m_min, m_break]
    ξ(m) = k2 m^{-α2} for m in (m_break, m_max]

    Continuous at m_break and normalized so:
      ∫_{m_min}^{m_max} m ξ(m) dm = 1  (one solar mass formed).

    This class exposes number-per-formed-mass in a mass interval:
      N([m0,m1]) per 1 Msun formed = ∫_{m0}^{m1} ξ(m) dm
    """

    alpha1: float
    alpha2: float
    m_min: float = 0.08
    m_break: float = 0.5
    m_max: float = 120.0

    def number_per_msun(self, m0: float, m1: float) -> float:
        m0 = max(m0, self.m_min)
        m1 = min(m1, self.m_max)
        if m1 <= m0:
            return 0.0

        k2_over_k1 = self.m_break ** (self.alpha2 - self.alpha1)

        mass_low = _powerlaw_mass_int(self.m_min, self.m_break, self.alpha1)
        mass_high = _powerlaw_mass_int(self.m_break, self.m_max, self.alpha2)
        k1 = 1.0 / (mass_low + k2_over_k1 * mass_high)
        k2 = k1 * k2_over_k1

        if m1 <= self.m_break:
            return k1 * _powerlaw_int(m0, m1, self.alpha1)
        if m0 >= self.m_break:
            return k2 * _powerlaw_int(m0, m1, self.alpha2)
        return k1 * _powerlaw_int(m0, self.m_break, self.alpha1) + k2 * _powerlaw_int(
            self.m_break, m1, self.alpha2
        )


def imf_preset(name: str) -> TwoPartIMF:
    """
    Simple presets for sensitivity studies.

    - kroupa: canonical high-mass slope α2=2.3
    - salpeter: α2=2.35
    - top-heavy: flatter high-mass slope
    - top-light: steeper high-mass slope
    """
    name = name.strip().lower()
    if name == "kroupa":
        return TwoPartIMF(alpha1=1.3, alpha2=2.3)
    if name == "salpeter":
        return TwoPartIMF(alpha1=1.3, alpha2=2.35)
    if name in {"top-heavy", "topheavy"}:
        return TwoPartIMF(alpha1=1.3, alpha2=1.9)
    if name in {"top-light", "toplight"}:
        return TwoPartIMF(alpha1=1.3, alpha2=2.7)
    raise ValueError(f"Unknown IMF preset: {name!r}")

