from __future__ import annotations

"""
PARSEC v2.0 VMS track parsing for *phase windows*.

Main idea
---------
From a stellar evolution track, we extract *durations* spent in:
- RSG conditions (toy definition: Teff and luminosity cut)
- carbon burning (toy definition: LC >= threshold)
- overlap (RSG AND carbon burning)

These durations are the only thing this repo needs from the track grid to do
IMF/SFR population estimates.

How to move toward neutrinos.pdf (Seong et al. 2025)
---------------------------------------------------
The paper uses MESA models and detailed neutrino emissivities/spectra.
To approach that:
- keep this phase-window extraction as the population backbone
- then calibrate neutrino luminosity/spectrum per phase using MESA runs or
  digitized/tabulated results from the paper
"""

import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MASS_RE = re.compile(r"_M(?P<mass>[0-9]+(?:\.[0-9]+)?)\.TAB$", re.IGNORECASE)


@dataclass(frozen=True)
class PhaseWindows:
    mass_msun: float

    lifetime_myr: float

    # Carbon burning: LC >= threshold
    cburn_start_myr: float | None
    cburn_end_myr: float | None
    cburn_duration_kyr: float
    cburn_max_lc: float | None

    # Red supergiant: Teff <= teff_max and logL>=logl_min
    rsg_duration_myr: float
    rsg_teff_max_k: float
    rsg_logl_min: float

    # Overlap: C-burning AND RSG
    cburn_rsg_duration_kyr: float

    lc_threshold: float
    zip_member: str


def find_track_members(zip_path: Path) -> dict[float, str]:
    members: dict[float, str] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            match = _MASS_RE.search(name)
            if not match:
                continue
            members[float(match.group("mass"))] = name
    return members


def _iter_data_rows(z: zipfile.ZipFile, member: str) -> tuple[list[str], list[list[str]]]:
    cols: list[str] | None = None
    rows: list[list[str]] = []
    with z.open(member) as f:
        for raw in f:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith("#"):
                continue
            if cols is None:
                cols = line.split()
                continue
            rows.append(line.split())
    if cols is None or not rows:
        raise RuntimeError(f"No data found in {member!r}")
    return cols, rows


def load_track_arrays(
    *,
    zip_path: Path,
    mass_msun: float,
    columns: tuple[str, ...] = ("AGE", "LOG_TE", "LOG_L", "LC"),
) -> dict[str, np.ndarray]:
    members = find_track_members(zip_path)
    if mass_msun not in members:
        raise FileNotFoundError(f"Mass {mass_msun:g} not in zip {zip_path}")
    member = members[mass_msun]
    with zipfile.ZipFile(zip_path) as z:
        cols, rows = _iter_data_rows(z, member)
    col_idx = {c: i for i, c in enumerate(cols)}
    missing = [c for c in columns if c not in col_idx]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {member!r}")
    out: dict[str, np.ndarray] = {}
    for c in columns:
        out[c] = np.array([float(r[col_idx[c]]) for r in rows], dtype=float)
    out["_zip_member"] = np.array([member])
    return out


def _segments_total_duration(age_yr: np.ndarray, mask: np.ndarray) -> tuple[float, float | None, float | None]:
    """
    Compute total time in `mask==True` (sum over contiguous segments).

    Returns: (total_duration_yr, first_start_age_yr, last_end_age_yr)
    """
    if age_yr.ndim != 1 or mask.ndim != 1 or age_yr.shape != mask.shape:
        raise ValueError("age_yr and mask must be 1D arrays of the same shape")
    if age_yr.size < 2:
        return 0.0, None, None

    total = 0.0
    start: float | None = None
    first_start: float | None = None
    last_end: float | None = None
    in_seg = False
    for i in range(age_yr.size):
        if bool(mask[i]) and not in_seg:
            in_seg = True
            start = float(age_yr[i])
            if first_start is None:
                first_start = start
        if (not bool(mask[i]) or i == age_yr.size - 1) and in_seg:
            end_age = float(age_yr[i]) if bool(mask[i]) and i == age_yr.size - 1 else float(age_yr[i - 1])
            if start is not None:
                total += max(0.0, end_age - start)
            last_end = end_age
            in_seg = False
            start = None
    return total, first_start, last_end


def extract_phase_windows(
    *,
    zip_path: Path,
    mass_msun: float,
    lc_threshold: float,
    rsg_teff_max_k: float,
    rsg_logl_min: float,
) -> PhaseWindows:
    members = find_track_members(zip_path)
    if mass_msun not in members:
        available = ", ".join(f"{m:g}" for m in sorted(members.keys())[:30])
        raise FileNotFoundError(
            f"Mass {mass_msun:g} not in zip. First available masses: {available} ..."
        )
    member = members[mass_msun]

    with zipfile.ZipFile(zip_path) as z:
        cols, rows = _iter_data_rows(z, member)
        col_idx = {c: i for i, c in enumerate(cols)}
        for req in ("AGE", "LOG_TE", "LOG_L", "LC"):
            if req not in col_idx:
                raise RuntimeError(f"Missing required column {req!r} in {member!r}")

        age_yr = np.array([float(r[col_idx["AGE"]]) for r in rows], dtype=float)
        log_te = np.array([float(r[col_idx["LOG_TE"]]) for r in rows], dtype=float)
        log_l = np.array([float(r[col_idx["LOG_L"]]) for r in rows], dtype=float)
        lc = np.array([float(r[col_idx["LC"]]) for r in rows], dtype=float)

    if not np.all(np.diff(age_yr) >= 0):
        raise RuntimeError(f"AGE is not monotonically increasing in {member!r}")

    lifetime_myr = float(age_yr[-1] / 1e6)

    cburn_mask = lc >= lc_threshold
    cburn_dur_yr, cburn_start, cburn_end = _segments_total_duration(age_yr, cburn_mask)
    cburn_max_lc = float(np.nanmax(lc)) if lc.size else None

    teff_k = 10.0**log_te
    rsg_mask = (teff_k <= rsg_teff_max_k) & (log_l >= rsg_logl_min)
    rsg_dur_yr, _, _ = _segments_total_duration(age_yr, rsg_mask)

    cburn_rsg_mask = cburn_mask & rsg_mask
    cburn_rsg_dur_yr, _, _ = _segments_total_duration(age_yr, cburn_rsg_mask)

    return PhaseWindows(
        mass_msun=float(mass_msun),
        lifetime_myr=lifetime_myr,
        cburn_start_myr=None if cburn_start is None else float(cburn_start / 1e6),
        cburn_end_myr=None if cburn_end is None else float(cburn_end / 1e6),
        cburn_duration_kyr=float(cburn_dur_yr / 1e3),
        cburn_max_lc=cburn_max_lc,
        rsg_duration_myr=float(rsg_dur_yr / 1e6),
        rsg_teff_max_k=float(rsg_teff_max_k),
        rsg_logl_min=float(rsg_logl_min),
        cburn_rsg_duration_kyr=float(cburn_rsg_dur_yr / 1e3),
        lc_threshold=float(lc_threshold),
        zip_member=member,
    )


def extract_phase_windows_to_csv(
    *,
    zip_path: Path,
    masses_msun: list[float],
    out_csv: Path,
    lc_threshold: float = 1e-3,
    rsg_teff_max_k: float = 4000.0,
    rsg_logl_min: float = 4.5,
) -> None:
    windows: list[PhaseWindows] = []
    for m in masses_msun:
        windows.append(
            extract_phase_windows(
                zip_path=zip_path,
                mass_msun=m,
                lc_threshold=lc_threshold,
                rsg_teff_max_k=rsg_teff_max_k,
                rsg_logl_min=rsg_logl_min,
            )
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mass_msun",
                "lifetime_myr",
                "cburn_start_myr",
                "cburn_end_myr",
                "cburn_duration_kyr",
                "cburn_max_lc",
                "rsg_duration_myr",
                "rsg_teff_max_k",
                "rsg_logl_min",
                "cburn_rsg_duration_kyr",
                "lc_threshold",
                "zip_member",
            ],
        )
        writer.writeheader()
        for w in sorted(windows, key=lambda x: x.mass_msun):
            writer.writerow(
                {
                    "mass_msun": f"{w.mass_msun:.1f}",
                    "lifetime_myr": f"{w.lifetime_myr:.6f}",
                    "cburn_start_myr": "" if w.cburn_start_myr is None else f"{w.cburn_start_myr:.6f}",
                    "cburn_end_myr": "" if w.cburn_end_myr is None else f"{w.cburn_end_myr:.6f}",
                    "cburn_duration_kyr": f"{w.cburn_duration_kyr:.3f}",
                    "cburn_max_lc": "" if w.cburn_max_lc is None else f"{w.cburn_max_lc:.6g}",
                    "rsg_duration_myr": f"{w.rsg_duration_myr:.6f}",
                    "rsg_teff_max_k": f"{w.rsg_teff_max_k:.6g}",
                    "rsg_logl_min": f"{w.rsg_logl_min:.6g}",
                    "cburn_rsg_duration_kyr": f"{w.cburn_rsg_duration_kyr:.3f}",
                    "lc_threshold": f"{w.lc_threshold:.6g}",
                    "zip_member": w.zip_member,
                }
            )

