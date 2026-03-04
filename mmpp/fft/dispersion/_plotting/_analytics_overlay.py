"""Helper for overlaying analytical dispersion curves on S(k,f) heatmaps.

Extracts material parameters from zarr simulation attributes and dispatches
to the appropriate model from :mod:`mmpp.analytical.dispersion`.

Internal module — used by :class:`DispersionPlotAccessor.add_analytics`.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from ..models import DispersionResult1D

logger = logging.getLogger(__name__)

# ── SW-config presets ─────────────────────────────────────────────────────
# Maps human-readable geometry names to phi angles (radians)
SW_CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    "DE":  {"phi": math.pi / 2, "label": "Damon-Eshbach (k⊥M)"},
    "BV":  {"phi": 0.0,         "label": "Backward Volume (k∥M)"},
    "FV":  {"phi": None,        "label": "Forward Volume (M⊥film)"},      # special model
    "MSSW": {"phi": math.pi / 2, "label": "MSSW (k⊥M)"},                  # alias for DE
}

# ── Model registry ────────────────────────────────────────────────────────
# Maps model names to callables from mmpp.analytical.dispersion
_MODEL_REGISTRY: dict[str, str] = {
    "kalinikos":       "kalinikos",
    "kalinikos_slavin": "kalinikos",
    "ks":              "kalinikos",
    "damon_eshbach":   "damon_eshbach",
    "de":              "damon_eshbach",
    "backward_volume": "backward_volume",
    "bv":              "backward_volume",
    "forward_volume":  "forward_volume",
    "fv":              "forward_volume",
    "bottcher":        "bottcher",
    "kim":             "kim",
    "cortes_ortuno":   "cortes_ortuno",
}


def _get_model_func(name: str) -> Callable:
    """Resolve model name to the actual function from analytical.dispersion."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    func_name = _MODEL_REGISTRY.get(key)
    if func_name is None:
        available = sorted(set(_MODEL_REGISTRY.values()))
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}"
        )
    from mmpp.analytical import dispersion as _ad
    func = getattr(_ad, func_name, None)
    if func is None:
        raise ImportError(f"Model function '{func_name}' not found in mmpp.analytical.dispersion")
    return func


def extract_material_params(result: "DispersionResult1D") -> dict[str, Any]:
    """Auto-extract material parameters from zarr attrs via DispersionResult1D back-reference.

    Traverses the chain: result._interface → parent_fft → job_result → zarr attrs.

    Returns
    -------
    dict
        Keys: ``B``, ``Ms``, ``d``, ``Aex``, ``Ku``, with ``None`` for missing values.
    """
    params: dict[str, Any] = {
        "B": None,
        "Ms": None,
        "d": None,
        "Aex": None,
        "Ku": None,
    }

    # Navigate back-reference chain
    iface = getattr(result, "_interface", None)
    if iface is None:
        logger.debug("No _interface back-reference on result; cannot auto-detect params")
        return params

    parent_fft = getattr(iface, "parent_fft", None)
    if parent_fft is None:
        return params

    job_result = getattr(parent_fft, "job_result", None)
    if job_result is None:
        return params

    # Get zarr attrs
    attrs: dict[str, Any] = {}
    try:
        import zarr
        root = zarr.open(job_result.path, mode="r")
        attrs = dict(root.attrs)
    except Exception as exc:
        logger.debug("Failed to read zarr attrs: %s", exc)
        return params

    # B_ext — might be scalar or array
    b_ext = attrs.get("B_ext")
    if b_ext is not None:
        try:
            b_val = np.asarray(b_ext, dtype=float)
            if b_val.ndim == 0:
                params["B"] = float(b_val)
            else:
                # Vector → magnitude
                params["B"] = float(np.linalg.norm(b_val))
        except (TypeError, ValueError):
            pass

    # Msat
    msat = attrs.get("Msat")
    if msat is not None:
        try:
            params["Ms"] = float(msat)
        except (TypeError, ValueError):
            pass

    # Aex
    aex = attrs.get("Aex")
    if aex is not None:
        try:
            params["Aex"] = float(aex)
        except (TypeError, ValueError):
            pass

    # Thickness = Nz * dz
    nz = attrs.get("Nz")
    dz = attrs.get("dz")
    if nz is not None and dz is not None:
        try:
            params["d"] = float(nz) * float(dz)
        except (TypeError, ValueError):
            pass

    # Anisotropy
    ku = attrs.get("Ku") or attrs.get("kc1") or attrs.get("Ku1")
    if ku is not None:
        try:
            params["Ku"] = float(ku)
        except (TypeError, ValueError):
            pass

    detected = {k: v for k, v in params.items() if v is not None}
    logger.info("Auto-detected material params from zarr: %s", detected)
    return params


def compute_analytical_dispersion(
    k_range: tuple[float, float],
    *,
    model: str = "kalinikos",
    sw_config: str = "DE",
    n_modes: int = 1,
    k_points: int = 500,
    phi: Optional[float] = None,
    D: Optional[float] = None,
    # Material params (required):
    B: float,
    Ms: float,
    d: float,
    Aex: float,
    Ku: float = 0.0,
    g: float = 2.0,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Compute analytical dispersion curve(s).

    Returns
    -------
    list of (k_array, f_ghz_array, label)
        One entry per mode.
    """
    k_arr = np.linspace(k_range[0], k_range[1], k_points)

    # Resolve sw_config → phi
    preset = SW_CONFIG_PRESETS.get(sw_config.upper(), {})
    effective_phi = phi if phi is not None else preset.get("phi", math.pi / 2)
    config_label = preset.get("label", sw_config)

    model_func = _get_model_func(model)
    func_name = model_func.__name__

    results: list[tuple[np.ndarray, np.ndarray, str]] = []

    for n in range(n_modes):
        kwargs: dict[str, Any] = {
            "k": k_arr,
            "B": B,
            "Ms": Ms,
            "d": d,
            "Aex": Aex,
            "g": g,
        }

        # Add Ku if the model accepts it
        if func_name not in ("forward_volume",):
            kwargs["Ku"] = Ku

        # Add phi for models that accept it
        if func_name in ("kalinikos",):
            kwargs["phi"] = effective_phi
        elif func_name == "cortes_ortuno":
            kwargs["phi"] = effective_phi
            if D is not None:
                kwargs["D"] = D

        # Add DMI for kim model
        if func_name == "kim" and D is not None:
            kwargs["D"] = D
            if phi is not None:
                kwargs["phi"] = phi

        # PSSW mode index (only for kalinikos_no_approx)
        if n > 0:
            # Switch to kalinikos_no_approx for n>0
            try:
                from mmpp.analytical.dispersion import kalinikos_no_approx
                kwargs["n"] = n
                if "phi" in kwargs:
                    del kwargs["phi"]  # kalinikos_no_approx doesn't take phi
                disp_result = kalinikos_no_approx(**kwargs)
            except (ImportError, TypeError) as exc:
                logger.warning("PSSW mode n=%d failed: %s", n, exc)
                continue
        else:
            try:
                disp_result = model_func(**kwargs)
            except TypeError as exc:
                # Model might not accept some kwargs
                logger.warning("Model %s call failed: %s — retrying with minimal params", func_name, exc)
                minimal = {k: v for k, v in kwargs.items() if k in ("k", "B", "Ms", "d", "Aex", "g")}
                disp_result = model_func(**minimal)

        mode_label = config_label if n == 0 else f"{config_label} (n={n})"
        results.append((disp_result.k, disp_result.f, mode_label))

    return results
