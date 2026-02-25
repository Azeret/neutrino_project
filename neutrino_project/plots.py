from __future__ import annotations

from pathlib import Path

import numpy as np

from .parsec_v2_vms import load_track_arrays


def _ensure_matplotlib_cache_dirs() -> None:
    """
    Avoid slow imports + noisy warnings when user home is not writable.
    """
    import os

    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    Path(os.environ.setdefault("MPLCONFIGDIR", str(tmp / "matplotlib"))).mkdir(parents=True, exist_ok=True)
    Path(os.environ.setdefault("XDG_CACHE_HOME", str(tmp / "xdg_cache"))).mkdir(parents=True, exist_ok=True)


def plot_parsec_hrd(
    *,
    zip_path: Path,
    mass_msun: float,
    out_png: Path,
    lc_threshold: float = 1e-3,
    rsg_teff_max_k: float = 4000.0,
    rsg_logl_min: float = 4.5,
) -> None:
    """
    Simple HR diagram (logTe vs logL) and highlight:
    - RSG region (toy criterion)
    - carbon burning (LC threshold)
    - overlap
    """
    track = load_track_arrays(zip_path=zip_path, mass_msun=mass_msun, columns=("LOG_TE", "LOG_L", "LC"))
    log_te = track["LOG_TE"]
    log_l = track["LOG_L"]
    lc = track["LC"]

    teff = 10.0**log_te
    rsg = (teff <= rsg_teff_max_k) & (log_l >= rsg_logl_min)
    cburn = lc >= lc_threshold
    both = rsg & cburn

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(log_te, log_l, color="0.75", lw=1.0, label="track")
    if rsg.any():
        ax.scatter(log_te[rsg], log_l[rsg], s=8, color="#d62728", label="RSG")
    if cburn.any():
        ax.scatter(log_te[cburn], log_l[cburn], s=10, color="#1f77b4", label="C-burning (LC)")
    if both.any():
        ax.scatter(log_te[both], log_l[both], s=14, color="#9467bd", label="overlap")

    ax.invert_xaxis()
    ax.set_xlabel(r"$\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$")
    ax.set_ylabel(r"$\log_{10}(L/L_\odot)$")
    ax.set_title(f"PARSEC v2.0 VMS track: {mass_msun:g} Msun")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_alpha_spectrum(
    *,
    mean_energy_mev: float,
    alpha: float,
    out_png: Path,
    e_min_mev: float = 0.0,
    e_max_mev: float = 5.0,
    n_grid: int = 2000,
) -> None:
    from .neutrinos import NeutrinoSpectrumModel

    spec = NeutrinoSpectrumModel.alpha_fit(mean_energy_mev=mean_energy_mev, alpha=alpha)
    e = np.linspace(e_min_mev, e_max_mev, int(n_grid))
    f = spec.pdf(e)
    if f.max() > 0:
        f = f / np.trapz(f, e)

    _ensure_matplotlib_cache_dirs()
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(e, f, color="black")
    ax.set_xlabel("Eν [MeV]")
    ax.set_ylabel("f(E) [MeV⁻¹] (normalized)")
    ax.set_title(f"Alpha-fit number spectrum (<E>={mean_energy_mev:g} MeV, α={alpha:g})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

