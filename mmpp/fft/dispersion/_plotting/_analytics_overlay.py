"""Helper for overlaying analytical dispersion curves on S(k,f) heatmaps.

Extracts material parameters from zarr simulation attributes and dispatches
to the appropriate model from :mod:`mmpp.analytical.dispersion`.

Internal module — used by :class:`DispersionPlotAccessor.add_analytics`.
"""

from __future__ import annotations

import ast
import glob
import logging
import math
import os
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..models import DispersionResult1D

logger = logging.getLogger(__name__)

_MX3_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=)\s*(.+?)\s*$")

# ── SW-config presets ─────────────────────────────────────────────────────
# Maps human-readable geometry names to phi angles (radians)
SW_CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    "DE": {"phi": math.pi / 2, "label": "Damon-Eshbach (k⊥M)"},
    "BV": {"phi": 0.0, "label": "Backward Volume (k∥M)"},
    "FV": {"phi": None, "label": "Forward Volume (M⊥film)"},  # special model
    "MSSW": {"phi": math.pi / 2, "label": "MSSW (k⊥M)"},  # alias for DE
}

# ── Model registry ────────────────────────────────────────────────────────
# Maps model names to callables from mmpp.analytical.dispersion
_MODEL_REGISTRY: dict[str, str] = {
    "kalinikos": "kalinikos",
    "kalinikos_slavin": "kalinikos",
    "ks": "kalinikos",
    "damon_eshbach": "damon_eshbach",
    "de": "damon_eshbach",
    "backward_volume": "backward_volume",
    "bv": "backward_volume",
    "forward_volume": "forward_volume",
    "fv": "forward_volume",
    "bottcher": "bottcher",
    "kim": "kim",
    "cortes_ortuno": "cortes_ortuno",
}


def _get_model_func(name: str) -> Callable:
    """Resolve model name to the actual function from analytical.dispersion."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    func_name = _MODEL_REGISTRY.get(key)
    if func_name is None:
        available = sorted(set(_MODEL_REGISTRY.values()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    from mmpp.analytical import dispersion as _ad

    func = getattr(_ad, func_name, None)
    if func is None:
        raise ImportError(
            f"Model function '{func_name}' not found in mmpp.analytical.dispersion"
        )
    return func


def _is_mumax_pointer(val: Any) -> bool:
    """Check if value is a mumax3 memory pointer string like '0xc00121ee70'."""
    if not isinstance(val, str):
        return False
    return val.startswith("0x") and len(val) > 4


def _safe_float(val: Any) -> float | None:
    """Convert value to float, handling strings and mumax3 pointers."""
    if val is None:
        return None
    if _is_mumax_pointer(val):
        return None  # Can't extract from pointer
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _sidecar_mx3_path(job_result: Any) -> str | None:
    """Return the best matching sidecar ``.mx3`` file for a zarr job."""
    if job_result is None or not getattr(job_result, "path", None):
        return None
    zarr_path = str(job_result.path)
    base_name = os.path.basename(zarr_path).replace(".zarr", "")
    parent_dir = os.path.dirname(zarr_path)
    exact = os.path.join(parent_dir, f"{base_name}.mx3")
    if os.path.exists(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(parent_dir, f"{base_name}.mx3*")))
    return matches[0] if matches else None


def _eval_mx3_expr(expr: str, env: Mapping[str, Any]) -> Any:
    """Evaluate a small, safe subset of mumax3 scalar/vector expressions."""
    tree = ast.parse(expr.strip().rstrip(";").replace("^", "**"), mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError("unsupported constant")
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
            if node.id in {"true", "True"}:
                return True
            if node.id in {"false", "False"}:
                return False
            raise ValueError(f"unknown name {node.id!r}")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("unsupported binary operator")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unsupported unary operator")
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(_eval(elt) for elt in node.elts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "vector":
                return tuple(_eval(arg) for arg in node.args)
            if name == "sin":
                return math.sin(_eval(node.args[0]))
            if name == "cos":
                return math.cos(_eval(node.args[0]))
            if name == "tan":
                return math.tan(_eval(node.args[0]))
            if name == "sqrt":
                return math.sqrt(_eval(node.args[0]))
            if name == "abs":
                return abs(_eval(node.args[0]))
            if name == "log":
                return math.log(_eval(node.args[0]))
            if name == "exp":
                return math.exp(_eval(node.args[0]))
            raise ValueError(f"unsupported call {name!r}")
        raise ValueError("unsupported syntax")

    return _eval(tree)


def _parse_mx3_scalars(path: str | None) -> dict[str, Any]:
    """Parse simple mumax3 assignments from a sidecar script."""
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return {}

    env: dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.split("//", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        match = _MX3_ASSIGNMENT_RE.match(line)
        if not match:
            continue
        name, expr = match.groups()
        try:
            env[name] = _eval_mx3_expr(expr, env)
        except Exception:
            continue
    return env


def _safe_field_vector(value: Any) -> tuple[float, float, float] | None:
    """Coerce scalar/vector field values into a 3D vector in Tesla."""
    if value is None or _is_mumax_pointer(value):
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 1:
        return (0.0, 0.0, float(arr[0]))
    if arr.size >= 3:
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    return None


def _field_magnitude(value: Any) -> float | None:
    vector = _safe_field_vector(value)
    if vector is None:
        return _safe_float(value)
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def _field_phi_for_axis(
    vector: tuple[float, float, float] | None,
    axis: str,
) -> float | None:
    """Return in-plane angle between propagation axis and static field."""
    if vector is None:
        return None
    field_xy = np.asarray(vector[:2], dtype=float)
    norm = float(np.linalg.norm(field_xy))
    if norm == 0.0:
        return None
    axis_key = str(axis or "x").lower()
    k_vec = np.array([1.0, 0.0]) if axis_key == "x" else np.array([0.0, 1.0])
    cos_phi = float(np.clip(np.dot(field_xy, k_vec) / norm, -1.0, 1.0))
    return float(math.acos(cos_phi))


def extract_material_params(result: DispersionResult1D) -> dict[str, Any]:
    """Auto-extract material parameters from zarr attrs via DispersionResult1D back-reference.

    Traverses the chain: result._interface → parent_fft → job_result → zarr attrs.

    Handles mumax3 quirks:
    - Pointer strings (``'0xc00...'``) for spatially-varying quantities
    - String-typed numbers (``'996000'`` instead of ``996000``)
    - Fallback to ``Bmax`` when ``B_ext`` is a pointer

    Returns
    -------
    dict
        Keys: ``B``, ``Ms``, ``d``, ``Aex``, ``Ku``, with ``None`` for missing values.
    """
    params: dict[str, Any] = {
        "B": None,
        "B_vector": None,
        "Ms": None,
        "d": None,
        "Aex": None,
        "phi": None,
        "Ku": 0.0,  # default to 0 — most sims don't have uniaxial anisotropy
    }

    # Navigate back-reference chain
    iface = getattr(result, "_interface", None)
    if iface is None:
        logger.debug(
            "No _interface back-reference on result; cannot auto-detect params"
        )
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
    mx3_path = _sidecar_mx3_path(job_result)
    mx3_params = _parse_mx3_scalars(mx3_path)

    # ── B_ext ─────────────────────────────────────────────────
    b_ext = attrs.get("B_ext")
    b_val = None

    if b_ext is not None and not _is_mumax_pointer(b_ext):
        b_val = _field_magnitude(b_ext)

    if b_val is None:
        # Fallback: try Bmax (common mumax3 scalar attr for bias field)
        bmax = _safe_float(attrs.get("Bmax"))
        if bmax is not None:
            b_val = abs(bmax)
            logger.info(
                "B_ext is a mumax3 pointer; using Bmax=%.4f T as fallback", b_val
            )
        else:
            # Try 'b' attr
            b_scalar = _safe_float(attrs.get("b"))
            if b_scalar is not None:
                b_val = abs(b_scalar)
                logger.info(
                    "B_ext is a mumax3 pointer; using |b|=%.4f T as fallback", b_val
                )

    b_vector = _safe_field_vector(b_ext)
    if b_vector is None:
        for key in ("B_ext", "Bext", "B", "B0", "Bmax", "bex", "b"):
            if key not in mx3_params:
                continue
            b_vector = _safe_field_vector(mx3_params[key])
            if b_vector is not None:
                b_val = _field_magnitude(b_vector)
                logger.info(
                    "Using static field from sidecar %s: %s=%s -> |B|=%.6g T",
                    mx3_path,
                    key,
                    mx3_params[key],
                    b_val,
                )
                break

    if b_val is None:
        logger.warning(
            "Cannot auto-detect B field (B_ext='%s' is a mumax3 pointer). "
            "Please provide B= manually or keep the matching .mx3 sidecar next to the .zarr.",
            b_ext,
        )
    params["B"] = b_val
    params["B_vector"] = b_vector
    phi = _field_phi_for_axis(b_vector, getattr(result, "axis", "x"))
    if phi is not None:
        params["phi"] = phi

    # ── Msat ──────────────────────────────────────────────────
    ms_val = _safe_float(attrs.get("Msat"))
    if ms_val is None:
        ms_val = _safe_float(mx3_params.get("Msat"))
    if ms_val is not None:
        params["Ms"] = ms_val
    else:
        logger.warning("Cannot auto-detect Ms (Msat='%s')", attrs.get("Msat"))

    # ── Aex ───────────────────────────────────────────────────
    aex_val = _safe_float(attrs.get("Aex"))
    if aex_val is None:
        aex_val = _safe_float(mx3_params.get("Aex"))
    if aex_val is not None:
        params["Aex"] = aex_val

    # ── Thickness ─────────────────────────────────────────────
    # Prefer Tz (total thickness) if available, otherwise Nz * dz
    tz = _safe_float(attrs.get("Tz"))
    if tz is None:
        tz = _safe_float(mx3_params.get("Tz"))
    if tz is not None and tz > 0:
        params["d"] = tz
    else:
        nz = _safe_float(attrs.get("Nz"))
        dz = _safe_float(attrs.get("dz"))
        if nz is not None and dz is not None:
            params["d"] = nz * dz

    # ── Anisotropy ────────────────────────────────────────────
    # Uniaxial
    ku_val = _safe_float(attrs.get("Ku")) or _safe_float(attrs.get("Ku1"))
    if ku_val is not None:
        params["Ku"] = ku_val

    # Cubic anisotropy — extract Kc1/Kc2 and phi_ani directly
    # (kalinikos now handles cubic natively, no Ku_eff approximation)
    kc1_val = _safe_float(attrs.get("kc1"))
    kc2_val = _safe_float(attrs.get("kc2"))
    if kc1_val is not None and kc1_val != 0.0:
        params["Kc1"] = kc1_val
        logger.info("Detected cubic anisotropy Kc1=%.1f J/m³", kc1_val)
    if kc2_val is not None and kc2_val != 0.0:
        params["Kc2"] = kc2_val

    # phi_ani from zarr (mumax3 convention: angle of anisC1 in-plane)
    phi_ani_val = _safe_float(attrs.get("phi_ani"))
    if phi_ani_val is not None:
        params["phi_ani"] = phi_ani_val
        logger.info("Detected cubic axis angle phi_ani=%.4f rad", phi_ani_val)

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
    phi: float | None = None,
    D: float | None = None,
    # Material params (required):
    B: float,
    Ms: float,
    d: float,
    Aex: float,
    Ku: float = 0.0,
    Kc1: float = 0.0,
    Kc2: float = 0.0,
    phi_ani: float = 0.0,
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
            # Pass cubic anisotropy if non-zero
            if abs(Kc1) > 0 or abs(Kc2) > 0:
                kwargs["Kc1"] = Kc1
                kwargs["Kc2"] = Kc2
                kwargs["phi_ani"] = phi_ani
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
                # Model might not accept some kwargs — retry preserving phi
                logger.warning(
                    "Model %s call failed: %s — retrying without Ku",
                    func_name,
                    exc,
                )
                # Keep phi but drop Ku (most common cause of TypeError)
                safe_keys = {"k", "B", "Ms", "d", "Aex", "g", "phi", "D"}
                minimal = {
                    k: v for k, v in kwargs.items() if k in safe_keys and v is not None
                }
                disp_result = model_func(**minimal)

        mode_label = config_label if n == 0 else f"{config_label} (n={n})"
        results.append((disp_result.k, disp_result.f, mode_label))

    return results
