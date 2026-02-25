"""
neutrino_project: a simple Milky Way massive-star population toy model.

Main goal (for now)
-------------------
Estimate the expected number of **carbon-burning red supergiants (C-burning RSGs)**
in the Milky Way and how that number changes with the IMF.

What this repo is / isn't
-------------------------
- Uses PARSEC v2.0 VMS tracks to estimate **phase durations** (RSG, C-burning, overlap).
- Uses a deliberately simple MW spatial model to estimate the fraction within a radius of the Sun.
- Includes a *toy* neutrino spectrum + cross section to get order-of-magnitude event rates.

It does NOT reproduce the detailed, MESA-based neutrino emissivities and detector simulations in:
  Seong et al. (2025), "Neutrinos from Carbon-Burning Red Supergiants and Their Detectability".
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

