"""Parameter resolution glue for analytical vortex models."""

from __future__ import annotations

import ast
import glob
import math
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmpp.analytical import (
    DiskGeometry,
    MaterialParams,
    omega0_novosad,
    reduce_mumax_slonczewski_cpp,
)

_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=)\s*(.+?)\s*$")

_ALIASES: dict[str, tuple[str, ...]] = {
    "Ms": ("Ms", "ms", "Msat"),
    "alpha": ("alpha",),
    "P": ("P", "pol", "polarization"),
    "Lambda": ("Lambda", "lambda_stt", "lambda"),
    "epsilonprime": ("epsilonprime", "epsilon_prime", "epsilonPrime"),
    "A": ("A", "Aex"),
    "R": ("R", "radius"),
    "L": ("L", "thickness", "th_pillar", "Tz"),
    "D": ("D", "diameter", "disk_diameter", "pillar_diameter"),
    "Area": ("Area", "area"),
    "omega0": ("omega0", "omega_0", "Omega0"),
    "N": ("N",),
    "domega0_dJ": ("domega0_dJ", "domega_dJ", "omega0_Oe_per_J"),
    "chi_scale": ("chi_scale",),
    "polarity": ("polarity", "p"),
    "field": ("field", "B", "B0", "Bext", "B_ext", "Bext_T"),
    "current_dir": ("current_dir", "currentDirection", "j_dir", "u_dir"),
    "current_density": ("Jdc", "J_dc", "current_density", "currentDensity", "J", "j"),
    "current": (
        "I_pillar_mA",
        "i_pillar_ma",
        "Idc_mA",
        "I_dc_mA",
        "current_mA",
        "I_pillar_A",
        "i_pillar_a",
        "Idc_A",
        "I_dc_A",
        "current_A",
        "i_pillar",
        "Idc",
        "I_dc",
        "i_dc",
        "current",
        "I",
    ),
    "Bx_T": ("Bx_T", "bx_t"),
    "By_T": ("By_T", "by_t"),
    "Bz_T": ("Bz_T", "bz_t"),
    "Bx_mT": ("Bx_mT", "bx_mt"),
    "By_mT": ("By_mT", "by_mt"),
    "Bz_mT": ("Bz_mT", "bz_mt"),
    "FixedLayer": ("FixedLayer", "fixed_layer", "polarizer"),
    "FixedLayerPosition": ("FixedLayerPosition", "fixed_layer_position"),
    "FreeLayerThickness": ("FreeLayerThickness", "free_layer_thickness"),
    "polarizer": ("polarizer", "fixed_layer_vector"),
    "p_z": ("p_z", "pz", "polarizer_z"),
    "mean_m_dot_p": (
        "mean_m_dot_p",
        "slonczewski_cos_theta",
        "mean_cos_theta",
    ),
    "x0": ("x0", "x0_m", "initial_x", "x_init", "core_x0"),
    "y0": ("y0", "y0_m", "initial_y", "y_init", "core_y0"),
    "x0_nm": ("x0_nm", "initial_x_nm"),
    "y0_nm": ("y0_nm", "initial_y_nm"),
    "dx": ("dx",),
    "dy": ("dy",),
    "dz": ("dz",),
    "Nx": ("Nx",),
    "Ny": ("Ny",),
    "Nz": ("Nz",),
}


@dataclass
class AnalyticalParameterResolution:
    """Resolved analytical-model inputs for vortex comparisons."""

    resolved_params: dict[str, Any]
    param_sources: dict[str, str]
    model_kind: str
    search_locations: tuple[str, ...] = field(default_factory=tuple)

    def get(self, key: str, default: Any = None) -> Any:
        return self.resolved_params.get(key, default)


def _mapping_from_view(mapping: Any) -> dict[str, Any]:
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        return dict(mapping)

    try:
        keys = list(mapping.keys())
    except Exception:
        try:
            return dict(mapping)
        except Exception:
            return {}

    out: dict[str, Any] = {}
    for key in keys:
        try:
            out[str(key)] = mapping[key]
        except Exception:
            continue
    return out


def _sidecar_mx3_path(job_result) -> str | None:
    if job_result is None or not getattr(job_result, "path", None):
        return None
    zarr_path = str(job_result.path)
    base_name = os.path.basename(zarr_path).replace(".zarr", "")
    parent_dir = os.path.dirname(zarr_path)
    matches = sorted(glob.glob(os.path.join(parent_dir, f"{base_name}.mx3*")))
    return matches[0] if matches else None


def _eval_mx3_expr(expr: str, env: Mapping[str, Any]) -> Any:
    expr_norm = expr.strip().rstrip(";").replace("^", "**")
    tree = ast.parse(expr_norm, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, (int, float, bool)):
                return value
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
            if node.id == "FIXEDLAYER_TOP":
                return "FIXEDLAYER_TOP"
            if node.id == "FIXEDLAYER_BOTTOM":
                return "FIXEDLAYER_BOTTOM"
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
            func_name = node.func.id
            if func_name == "vector":
                return tuple(_eval(arg) for arg in node.args)
            if func_name == "sin":
                return math.sin(_eval(node.args[0]))
            if func_name == "cos":
                return math.cos(_eval(node.args[0]))
            if func_name == "tan":
                return math.tan(_eval(node.args[0]))
            if func_name == "sqrt":
                return math.sqrt(_eval(node.args[0]))
            if func_name == "abs":
                return abs(_eval(node.args[0]))
            if func_name == "log":
                return math.log(_eval(node.args[0]))
            if func_name == "exp":
                return math.exp(_eval(node.args[0]))
            raise ValueError(f"unsupported call {func_name!r}")
        raise ValueError("unsupported syntax")

    return _eval(tree)


def _parse_mx3_scalars(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    env: dict[str, Any] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return {}

    for raw_line in lines:
        line = raw_line.split("//", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        match = _ASSIGNMENT_RE.match(line)
        if not match:
            continue
        name, expr = match.groups()
        try:
            env[name] = _eval_mx3_expr(expr, env)
        except Exception:
            continue
    return env


def _lookup_value(
    mapping: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> tuple[Any, str] | None:
    for alias in aliases:
        if alias in mapping:
            return mapping[alias], alias

    lower_map: dict[str, list[str]] = {}
    for key in mapping.keys():
        lower_map.setdefault(str(key).lower(), []).append(str(key))

    for alias in aliases:
        alias_str = str(alias)
        if len(alias_str) <= 1:
            continue
        keys = lower_map.get(alias_str.lower(), [])
        if not keys:
            continue
        if len(keys) == 1:
            return mapping[keys[0]], str(keys[0])
        for key in keys:
            if len(key) == len(alias_str):
                return mapping[key], str(key)
    return None


def _coerce_scalar(value: Any) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise TypeError("expected scalar value")
    return float(arr.reshape(-1)[0])


def _coerce_current_dir(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "x":
            return (1.0, 0.0)
        if token == "y":
            return (0.0, 1.0)
        if token == "-x":
            return (-1.0, 0.0)
        if token == "-y":
            return (0.0, -1.0)
        if token.startswith("(") and token.endswith(")"):
            value = ast.literal_eval(token)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError("current_dir must contain at least two components")
    return (float(arr[0]), float(arr[1]))


def _coerce_current_density(
    value: Any,
    *,
    source_key: str,
    area: float | None,
) -> float | Any:
    if callable(value):
        return value

    raw = _coerce_scalar(value)
    token = source_key.strip().lower()
    if "density" in token or token.startswith("j"):
        return raw

    current_a = raw
    if "ma" in token:
        current_a *= 1e-3
    elif "ua" in token:
        current_a *= 1e-6
    elif "na" in token:
        current_a *= 1e-9
    elif "pa" in token:
        current_a *= 1e-12

    if area is None or area <= 0.0:
        raise ValueError(
            "Current source was resolved as electrical current, but disk area "
            "could not be inferred to convert it to current density."
        )
    return current_a / area


def _resolve_aliases(
    canonical: str,
    param_keys: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    aliases: list[str] = []
    if param_keys:
        direct = param_keys.get(canonical)
        if direct is None and canonical == "current_density":
            direct = param_keys.get("current")
        if isinstance(direct, str):
            aliases.append(direct)
        elif isinstance(direct, (list, tuple)):
            aliases.extend(str(item) for item in direct)
    aliases.extend(_ALIASES.get(canonical, ()))
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        lowered = str(alias).lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(str(alias))
    return tuple(deduped)


def _resolve_polarity(job_result, trajectory=None) -> int:
    if trajectory is not None:
        values = np.asarray(getattr(trajectory, "polarity", []), dtype=float)
        if values.size:
            return 1 if float(np.mean(values)) >= 0.0 else -1
    attrs = _mapping_from_view(getattr(job_result, "attrs", None))
    hit = _lookup_value(attrs, _ALIASES["polarity"])
    if hit is None:
        return 1
    return 1 if _coerce_scalar(hit[0]) >= 0.0 else -1


def _derive_geometry(
    resolved: dict[str, Any],
    sources: dict[str, str],
    *,
    job_result,
    dataset_name: str | None,
) -> None:
    if "R" not in resolved:
        if "D" in resolved:
            resolved["R"] = 0.5 * _coerce_scalar(resolved["D"])
            sources["R"] = f"{sources['D']} -> R"
        elif "Area" in resolved:
            resolved["R"] = math.sqrt(
                max(_coerce_scalar(resolved["Area"]), 0.0) / math.pi
            )
            sources["R"] = f"{sources['Area']} -> R"
        else:
            attrs = _mapping_from_view(getattr(job_result, "attrs", None))
            dx = attrs.get("dx")
            dy = attrs.get("dy", dx)
            if dx is not None and dy is not None and dataset_name:
                try:
                    dataset = getattr(job_result, dataset_name)
                    shape = tuple(getattr(dataset, "shape", ()))
                    if len(shape) >= 4:
                        nx = int(shape[-2])
                        ny = int(shape[-3])
                        resolved["R"] = 0.5 * min(float(dx) * nx, float(dy) * ny)
                        sources["R"] = "dataset_shape+cell (full-box assumption)"
                        warnings.warn(
                            "Disk radius is absent from resolved metadata; assuming "
                            "the disk fills the smaller simulation-box dimension. "
                            "Provide R, D, or Area for quantitative Thiele predictions.",
                            UserWarning,
                            stacklevel=2,
                        )
                except Exception:
                    pass

    if "L" not in resolved:
        if (
            "FreeLayerThickness" in resolved
            and abs(_coerce_scalar(resolved["FreeLayerThickness"])) > 0.0
        ):
            resolved["L"] = _coerce_scalar(resolved["FreeLayerThickness"])
            sources["L"] = f"{sources['FreeLayerThickness']} -> L"
        elif "dz" in resolved and "Nz" in resolved:
            resolved["L"] = _coerce_scalar(resolved["dz"]) * max(
                int(_coerce_scalar(resolved["Nz"])), 1
            )
            sources["L"] = f"{sources['dz']}+{sources['Nz']} -> L"
        else:
            attrs = _mapping_from_view(getattr(job_result, "attrs", None))
            if "dz" in attrs:
                nz = float(attrs.get("Nz", 1.0))
                resolved["L"] = float(attrs["dz"]) * max(nz, 1.0)
                sources["L"] = "attrs:dz/Nz -> L"

    if "Area" not in resolved and "R" in resolved:
        resolved["Area"] = math.pi * _coerce_scalar(resolved["R"]) ** 2
        sources["Area"] = f"{sources['R']} -> Area"


def _derive_initial_displacement(
    resolved: dict[str, Any],
    sources: dict[str, str],
) -> None:
    if "x0" not in resolved and "x0_nm" in resolved:
        resolved["x0"] = _coerce_scalar(resolved["x0_nm"]) * 1e-9
        sources["x0"] = f"{sources['x0_nm']} -> x0"
    if "y0" not in resolved and "y0_nm" in resolved:
        resolved["y0"] = _coerce_scalar(resolved["y0_nm"]) * 1e-9
        sources["y0"] = f"{sources['y0_nm']} -> y0"


def _apply_source_layer(
    resolved: dict[str, Any],
    sources: dict[str, str],
    layer_name: str,
    mapping: Mapping[str, Any],
    *,
    param_keys: Mapping[str, Any] | None,
) -> None:
    for canonical in (
        "Ms",
        "alpha",
        "P",
        "Lambda",
        "epsilonprime",
        "A",
        "R",
        "L",
        "D",
        "Area",
        "current_density",
        "omega0",
        "N",
        "domega0_dJ",
        "chi_scale",
        "polarity",
        "field",
        "FixedLayer",
        "FixedLayerPosition",
        "FreeLayerThickness",
        "polarizer",
        "p_z",
        "x0",
        "y0",
        "x0_nm",
        "y0_nm",
        "Bx_T",
        "By_T",
        "Bz_T",
        "Bx_mT",
        "By_mT",
        "Bz_mT",
        "current",
        "current_dir",
        "dx",
        "dy",
        "dz",
        "Nx",
        "Ny",
        "Nz",
    ):
        if canonical in resolved:
            continue
        hit = _lookup_value(mapping, _resolve_aliases(canonical, param_keys))
        if hit is None:
            continue
        value, source_key = hit
        if canonical == "current_dir":
            try:
                value = _coerce_current_dir(value)
            except Exception:
                continue
        resolved[canonical] = value
        sources[canonical] = f"{layer_name}:{source_key}"


def _resolve_current_density(
    resolved: dict[str, Any],
    sources: dict[str, str],
    *,
    attrs_map: Mapping[str, Any],
    mx3_map: Mapping[str, Any],
    param_keys: Mapping[str, Any] | None,
    current: Any,
) -> None:
    if "current_density" in resolved:
        return

    area = None
    if "Area" in resolved:
        area = _coerce_scalar(resolved["Area"])

    deferred_current_hit: tuple[Any, str, str] | None = None
    if isinstance(current, str):
        for layer_name, mapping in (("attrs", attrs_map), ("mx3", mx3_map)):
            hit = _lookup_value(mapping, (current,))
            if hit is None:
                continue
            value, source_key = hit
            source_token = str(source_key).strip().lower()
            if "density" in source_token or source_token.startswith("j"):
                try:
                    resolved["current_density"] = _coerce_current_density(
                        value,
                        source_key=source_key,
                        area=area,
                    )
                except (TypeError, ValueError):
                    continue
                sources["current_density"] = f"{layer_name}:{source_key}"
                return
            deferred_current_hit = (value, source_key, layer_name)
            break

    if callable(current) or isinstance(current, (int, float)):
        resolved["current_density"] = current
        sources["current_density"] = "override:current"
        return

    for layer_name, mapping in (("attrs", attrs_map), ("mx3", mx3_map)):
        for alias in _resolve_aliases("current_density", param_keys):
            hit = _lookup_value(mapping, (alias,))
            if hit is None:
                continue
            value, source_key = hit
            try:
                resolved["current_density"] = _coerce_current_density(
                    value,
                    source_key=source_key,
                    area=area,
                )
            except (TypeError, ValueError):
                continue
            sources["current_density"] = f"{layer_name}:{source_key}"
            return

    if deferred_current_hit is not None:
        value, source_key, layer_name = deferred_current_hit
        try:
            resolved["current_density"] = _coerce_current_density(
                value,
                source_key=source_key,
                area=area,
            )
        except (TypeError, ValueError):
            return
        sources["current_density"] = f"{layer_name}:{source_key}"
        return

    for layer_name, mapping in (("attrs", attrs_map), ("mx3", mx3_map)):
        for alias in _resolve_aliases("current", param_keys):
            hit = _lookup_value(mapping, (alias,))
            if hit is None:
                continue
            value, source_key = hit
            try:
                resolved["current_density"] = _coerce_current_density(
                    value,
                    source_key=source_key,
                    area=area,
                )
            except (TypeError, ValueError):
                continue
            sources["current_density"] = f"{layer_name}:{source_key}"
            return


def _coerce_current_to_ampere(
    value: Any,
    *,
    source_key: str,
) -> float | Any:
    if callable(value):
        return value

    raw = _coerce_scalar(value)
    token = source_key.strip().lower()

    if "ma" in token:
        return raw * 1e-3
    if "ua" in token:
        return raw * 1e-6
    if "na" in token:
        return raw * 1e-9
    if "pa" in token:
        return raw * 1e-12
    return raw


def _current_source_has_explicit_unit(source_key: str) -> bool:
    token = str(source_key).strip().lower()
    return any(
        hint in token
        for hint in (
            "_ma",
            "ma",
            "_ua",
            "ua",
            "_na",
            "na",
            "_pa",
            "pa",
            "_a",
            "current_a",
        )
    )


def _source_precedence(source: str | None) -> int:
    token = str(source or "").strip().lower()
    if token.startswith("override:"):
        return 0
    if token.startswith("params:"):
        return 1
    if token.startswith("attrs:"):
        return 2
    if token.startswith("mx3:"):
        return 3
    if "-> current" in token or token.startswith("derived"):
        return 4
    if token.startswith("default"):
        return 5
    return 6


def _derived_current_from_density(
    resolved: Mapping[str, Any],
    sources: dict[str, str],
) -> tuple[float, float, str] | None:
    if "current_density" not in resolved or "Area" not in resolved:
        return None
    try:
        current_amp = _coerce_scalar(resolved["current_density"]) * _coerce_scalar(
            resolved["Area"]
        )
    except Exception:
        return None
    source = f"{sources.get('current_density', 'current_density')}+{sources.get('Area', 'Area')} -> current"
    return (float(current_amp), float(current_amp) * 1e3, source)


def _resolve_current_views(
    resolved: dict[str, Any],
    sources: dict[str, str],
    *,
    attrs_map: Mapping[str, Any],
    mx3_map: Mapping[str, Any],
    param_keys: Mapping[str, Any] | None,
    current: Any,
) -> None:
    if "current" not in resolved:
        if isinstance(current, str):
            for layer_name, mapping in (("attrs", attrs_map), ("mx3", mx3_map)):
                hit = _lookup_value(mapping, (current,))
                if hit is None:
                    continue
                value, source_key = hit
                resolved["current"] = value
                sources["current"] = f"{layer_name}:{source_key}"
                break
        elif isinstance(current, (int, float)) or callable(current):
            resolved["current"] = current
            sources["current"] = "override:current"

    if "current" not in resolved:
        for layer_name, mapping in (("attrs", attrs_map), ("mx3", mx3_map)):
            for alias in _resolve_aliases("current", param_keys):
                hit = _lookup_value(mapping, (alias,))
                if hit is None:
                    continue
                value, source_key = hit
                resolved["current"] = value
                sources["current"] = f"{layer_name}:{source_key}"
                break
            if "current" in resolved:
                break

    current_amp = None
    current_mA = None
    current_source = sources.get("current")
    if "current" in resolved:
        try:
            current_amp = _coerce_current_to_ampere(
                resolved["current"],
                source_key=sources.get("current", "current"),
            )
            if not callable(current_amp):
                current_mA = float(current_amp) * 1e3
        except Exception:
            current_amp = None
            current_mA = None
    derived = _derived_current_from_density(resolved, sources)
    if derived is not None:
        derived_amp, derived_mA, derived_source = derived
        density_source = sources.get("current_density", "")
        use_derived = current_amp is None or not _current_source_has_explicit_unit(
            current_source or ""
        )
        if not use_derived and _source_precedence(density_source) < _source_precedence(
            current_source
        ):
            use_derived = True
        if use_derived:
            current_amp = derived_amp
            current_mA = derived_mA
            sources["current"] = derived_source
            current_source = derived_source

    if current_amp is not None:
        resolved["current_A"] = current_amp
        sources["current_A"] = sources.get(
            "current", sources.get("current_density", "derived")
        )
    if current_mA is not None:
        resolved["current_mA"] = current_mA
        sources["current_mA"] = sources.get(
            "current", sources.get("current_density", "derived")
        )


def _resolve_field(
    resolved: dict[str, Any],
    sources: dict[str, str],
) -> None:
    if "field" in resolved:
        value = resolved["field"]
        if value is None or callable(value) or isinstance(value, (int, float)):
            return
        if isinstance(value, (tuple, list, np.ndarray)):
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size >= 3:
                return
        resolved.pop("field", None)
        sources.pop("field", None)

    if any(
        key in resolved for key in ("Bx_T", "By_T", "Bz_T", "Bx_mT", "By_mT", "Bz_mT")
    ):
        bx = (
            float(_coerce_scalar(resolved.get("Bx_T", 0.0)))
            if "Bx_T" in resolved
            else 0.0
        )
        by = (
            float(_coerce_scalar(resolved.get("By_T", 0.0)))
            if "By_T" in resolved
            else 0.0
        )
        bz = (
            float(_coerce_scalar(resolved.get("Bz_T", 0.0)))
            if "Bz_T" in resolved
            else 0.0
        )
        if "Bx_mT" in resolved:
            bx += float(_coerce_scalar(resolved["Bx_mT"])) * 1e-3
        if "By_mT" in resolved:
            by += float(_coerce_scalar(resolved["By_mT"])) * 1e-3
        if "Bz_mT" in resolved:
            bz += float(_coerce_scalar(resolved["Bz_mT"])) * 1e-3

        resolved["field"] = (bx, by, bz)
        field_sources = []
        for name in ("Bx_T", "By_T", "Bz_T", "Bx_mT", "By_mT", "Bz_mT"):
            if name in sources:
                field_sources.append(sources[name])
        sources["field"] = (
            " + ".join(field_sources) + " -> field"
            if field_sources
            else "derived:field_components"
        )


def _coerce_fixed_layer_position(value: Any) -> tuple[str, float]:
    token = str(value).strip().lower()
    if token in {"fixedlayer_top", "top", "+1", "1"}:
        return ("top", 1.0)
    if token in {"fixedlayer_bottom", "bottom", "-1", "2"}:
        return ("bottom", -1.0)
    raise ValueError("FixedLayerPosition must be TOP or BOTTOM")


def _coerce_polarizer(value: Any) -> tuple[float, float, float]:
    arr: Any = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 2:
        arr = np.array([arr[0], arr[1], 0.0], dtype=float)
    if arr.size < 3:
        raise ValueError("FixedLayer/polarizer must provide 2 or 3 components")
    vec = arr[:3]
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("FixedLayer/polarizer cannot be a zero vector")
    vec = vec / norm
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _resolve_cpp_spin_torque_terms(
    resolved: dict[str, Any],
    sources: dict[str, str],
) -> None:
    required = ("Ms", "alpha", "P")
    if not all(key in resolved for key in required):
        return

    if "L_stt" in resolved:
        torque_thickness = float(_coerce_scalar(resolved["L_stt"]))
    elif (
        "FreeLayerThickness" in resolved
        and abs(float(_coerce_scalar(resolved["FreeLayerThickness"]))) > 0.0
    ):
        torque_thickness = float(_coerce_scalar(resolved["FreeLayerThickness"]))
        resolved["L_stt"] = torque_thickness
        sources["L_stt"] = sources.get(
            "FreeLayerThickness", "derived:FreeLayerThickness"
        )
    elif "L" in resolved:
        torque_thickness = float(_coerce_scalar(resolved["L"]))
        resolved["L_stt"] = torque_thickness
        sources["L_stt"] = sources.get("L", "derived:L")
    else:
        return

    # If the caller explicitly provided analytical P in params without any
    # MuMax Slonczewski descriptors, treat it as a direct Guslienko-style
    # polarization. This lets manual ``params={...}`` bypass hidden mx3/attrs
    # reduction and use the exact values supplied by the user.
    _SLON_KEYS = (
        "polarizer",
        "FixedLayer",
        "p_z",
        "Lambda",
        "epsilonprime",
        "FixedLayerPosition",
        "mean_m_dot_p",
    )
    explicit_manual_p = sources.get("P", "").startswith("params:")
    explicit_manual_slonczewski = any(
        sources.get(key, "").startswith("params:") for key in _SLON_KEYS
    )

    if explicit_manual_p and not explicit_manual_slonczewski:
        has_explicit_slonczewski = False
    else:
        # Check whether any Slonczewski-specific parameter was explicitly
        # provided from any source (params/attrs/mx3, not from a default).
        # If none were provided, the user intends P as the direct Guslienko
        # spin-polarization.  Skip the MuMax reduction because its angle,
        # fixed-layer position and field-like term are otherwise undefined.
        has_explicit_slonczewski = any(
            key in resolved and not sources.get(key, "").startswith("default")
            for key in _SLON_KEYS
        )

    if not has_explicit_slonczewski:
        # Direct Guslienko mode: P_model = P (no Slonczewski reduction)
        raw_p = float(_coerce_scalar(resolved["P"]))
        resolved["P_raw"] = raw_p
        sources["P_raw"] = sources.get("P", "resolved:P")
        resolved["P_model"] = raw_p
        sources["P_model"] = (
            f"{sources.get('P', 'resolved:P')} -> P_model (direct, no Slonczewski)"
        )
        resolved["polarizer"] = (0.0, 0.0, 1.0)
        sources["polarizer"] = "default"
        resolved["p_z"] = 1.0
        sources["p_z"] = "default -> p_z"
        resolved["FixedLayerPosition"] = "FIXEDLAYER_TOP"
        sources["FixedLayerPosition"] = "default"
        resolved["fixed_layer_position"] = "top"
        sources["fixed_layer_position"] = "default"
        resolved["slonczewski_current_sign"] = 1.0
        sources["slonczewski_current_sign"] = "default -> sign"
        resolved["Lambda"] = 1.0
        sources["Lambda"] = "default"
        resolved["epsilonprime"] = 0.0
        sources["epsilonprime"] = "default"
        resolved["mean_m_dot_p"] = 0.0
        sources["mean_m_dot_p"] = "default:centered_vortex"
        if "domega0_dJ" not in resolved:
            resolved["domega0_dJ"] = 0.0
            sources["domega0_dJ"] = "default:0.0"
        return

    polarizer = None
    if "polarizer" in resolved:
        try:
            polarizer = _coerce_polarizer(resolved["polarizer"])
        except Exception:
            polarizer = None
    if polarizer is None and "FixedLayer" in resolved:
        try:
            polarizer = _coerce_polarizer(resolved["FixedLayer"])
            resolved["polarizer"] = polarizer
            sources["polarizer"] = sources.get("FixedLayer", "derived:FixedLayer")
        except Exception:
            polarizer = None
    if polarizer is None and "p_z" in resolved:
        pz = float(np.clip(_coerce_scalar(resolved["p_z"]), -1.0, 1.0))
        planar = math.sqrt(max(1.0 - pz * pz, 0.0))
        polarizer = (planar, 0.0, pz)
        resolved["polarizer"] = polarizer
        sources["polarizer"] = sources.get("p_z", "derived:p_z")
    if polarizer is None:
        polarizer = (0.0, 0.0, 1.0)
        resolved["polarizer"] = polarizer
        sources["polarizer"] = "default"

    if "p_z" not in resolved:
        resolved["p_z"] = float(polarizer[2])
        sources["p_z"] = f"{sources['polarizer']} -> p_z"

    if "FixedLayerPosition" in resolved:
        fixed_layer_position, current_sign = _coerce_fixed_layer_position(
            resolved["FixedLayerPosition"]
        )
    else:
        fixed_layer_position, current_sign = ("top", 1.0)
        resolved["FixedLayerPosition"] = "FIXEDLAYER_TOP"
        sources["FixedLayerPosition"] = "default"

    resolved["fixed_layer_position"] = fixed_layer_position
    sources["fixed_layer_position"] = sources.get("FixedLayerPosition", "default")
    resolved["slonczewski_current_sign"] = current_sign
    sources["slonczewski_current_sign"] = f"{sources['fixed_layer_position']} -> sign"

    if "Lambda" not in resolved:
        resolved["Lambda"] = 1.0
        sources["Lambda"] = "default"
    if "epsilonprime" not in resolved:
        resolved["epsilonprime"] = 0.0
        sources["epsilonprime"] = "default"
    if "mean_m_dot_p" not in resolved:
        resolved["mean_m_dot_p"] = 0.0
        sources["mean_m_dot_p"] = "default:centered_vortex"

    resolved["P_raw"] = float(_coerce_scalar(resolved["P"]))
    sources["P_raw"] = sources.get("P", "resolved:P")

    material = MaterialParams(
        Ms=float(_coerce_scalar(resolved["Ms"])),
        alpha=float(_coerce_scalar(resolved["alpha"])),
        P=float(_coerce_scalar(resolved["P"])),
        A=float(_coerce_scalar(resolved.get("A", 1.3e-11))),
    )
    reduction = reduce_mumax_slonczewski_cpp(
        material=material,
        torque_thickness=torque_thickness,
        polarizer=polarizer,
        fixed_layer_position=fixed_layer_position,
        Lambda=float(_coerce_scalar(resolved["Lambda"])),
        epsilonprime=float(_coerce_scalar(resolved["epsilonprime"])),
        mean_m_dot_p=float(_coerce_scalar(resolved["mean_m_dot_p"])),
    )

    resolved["P_eff"] = reduction.epsilon
    sources["P_eff"] = (
        f"{sources['P']}+{sources['Lambda']}+{sources['mean_m_dot_p']} -> P_eff"
    )
    resolved["P_model"] = reduction.pump_polarization
    sources["P_model"] = (
        f"{sources['P_eff']}+{sources['alpha']}+{sources['epsilonprime']}+"
        f"{sources['fixed_layer_position']} -> P_model"
    )
    resolved["phase_polarization"] = reduction.phase_polarization
    sources["phase_polarization"] = (
        f"{sources['P_eff']}+{sources['alpha']}+{sources['epsilonprime']}+"
        f"{sources['fixed_layer_position']} -> phase_polarization"
    )
    resolved["domega0_dJ_stt"] = reduction.phase_omega_per_J
    sources["domega0_dJ_stt"] = (
        f"{sources['phase_polarization']}+{sources['L_stt']}+{sources['Ms']} -> domega0_dJ_stt"
    )

    raw_domega = (
        float(_coerce_scalar(resolved["domega0_dJ"]))
        if "domega0_dJ" in resolved
        else 0.0
    )
    raw_source = sources.get("domega0_dJ", "default:0.0")
    resolved["domega0_dJ_user"] = raw_domega
    sources["domega0_dJ_user"] = raw_source
    resolved["domega0_dJ"] = raw_domega + reduction.phase_omega_per_J
    sources["domega0_dJ"] = f"{raw_source}+{sources['domega0_dJ_stt']} -> domega0_dJ"


def _select_model_kind(model: str, resolved: Mapping[str, Any]) -> str:
    token = str(model).strip().lower()
    if token in {"cpp", "cip"}:
        return token

    has_current = "current_density" in resolved
    has_current_dir = resolved.get("current_dir") is not None
    if has_current and has_current_dir:
        return "cip"
    if has_current:
        return "cpp"
    raise ValueError(
        "Could not infer whether analytical comparison should use CPP or CIP. "
        "Provide `model='cpp'` or `model='cip'`, or resolve current/current_dir explicitly."
    )


def extract_model_defaults(
    *,
    vortex_interface=None,
    job_result=None,
    dataset_name: str | None = None,
    slice_info=None,
    trajectory=None,
    params: str | Mapping[str, Any] = "auto",
    model: str = "auto",
    current: Any = None,
    param_keys: Mapping[str, Any] | None = None,
    **overrides,
) -> AnalyticalParameterResolution:
    """Resolve analytical-model parameters from attrs, sidecar ``.mx3``, and overrides."""
    _ = slice_info  # reserved for future coupled extraction logic

    if vortex_interface is not None:
        job_result = job_result or getattr(vortex_interface, "_job", None)
        dataset_name = dataset_name or getattr(vortex_interface, "_dataset", None)

    if job_result is None:
        raise ValueError(
            "job_result or vortex_interface is required for parameter extraction"
        )

    attrs_map = _mapping_from_view(getattr(job_result, "attrs", None))
    mx3_path = _sidecar_mx3_path(job_result)
    mx3_map = _parse_mx3_scalars(mx3_path)

    if isinstance(params, Mapping):
        params_map = dict(params)
        search_locations: tuple[str, ...] = (
            "params",
            "attrs",
            f"mx3:{mx3_path}" if mx3_path else "mx3:none",
            "overrides",
        )
    elif str(params).strip().lower() == "auto":
        params_map = {}
        search_locations = (
            "attrs",
            f"mx3:{mx3_path}" if mx3_path else "mx3:none",
            "overrides",
        )
    else:
        raise ValueError("params must be 'auto' or a mapping of parameter values")

    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {}

    if params_map:
        _apply_source_layer(
            resolved, sources, "params", params_map, param_keys=param_keys
        )
    _apply_source_layer(resolved, sources, "attrs", attrs_map, param_keys=param_keys)
    _apply_source_layer(resolved, sources, "mx3", mx3_map, param_keys=param_keys)

    _derive_geometry(
        resolved,
        sources,
        job_result=job_result,
        dataset_name=dataset_name,
    )
    _derive_initial_displacement(resolved, sources)
    _resolve_current_density(
        resolved,
        sources,
        attrs_map=attrs_map,
        mx3_map=mx3_map,
        param_keys=param_keys,
        current=current,
    )
    _resolve_current_views(
        resolved,
        sources,
        attrs_map=attrs_map,
        mx3_map=mx3_map,
        param_keys=param_keys,
        current=current,
    )
    _resolve_field(resolved, sources)

    if "polarity" not in resolved:
        resolved["polarity"] = _resolve_polarity(job_result, trajectory=trajectory)
        sources["polarity"] = "trajectory" if trajectory is not None else "default"

    if "chi_scale" not in resolved:
        resolved["chi_scale"] = 1.0
        sources["chi_scale"] = "default"

    for key, value in list(overrides.items()):
        if value is None:
            continue
        canonical = "current_density" if key == "current" else key
        if canonical == "current_dir":
            value = _coerce_current_dir(value)
        if canonical in {"polarizer", "FixedLayer"}:
            value = _coerce_polarizer(value)
        resolved[canonical] = value
        sources[canonical] = f"override:{key}"

    common_required = ("Ms", "alpha", "P", "R", "L", "current_density")
    model_kind = _select_model_kind(model, resolved)
    if model_kind == "cpp":
        _resolve_cpp_spin_torque_terms(resolved, sources)
    required = list(common_required)
    if model_kind == "cip":
        required.append("current_dir")

    missing = [key for key in required if key not in resolved]
    if missing:
        searched = " -> ".join(search_locations)
        raise ValueError(
            "Missing analytical parameters: "
            + ", ".join(missing)
            + f". Looked in {searched}. "
            + "Pass them explicitly as keyword arguments or via `param_keys`/`current`."
        )

    if "omega0" not in resolved:
        material = MaterialParams(
            Ms=_coerce_scalar(resolved["Ms"]),
            alpha=_coerce_scalar(resolved["alpha"]),
            P=_coerce_scalar(resolved["P"]),
            A=_coerce_scalar(resolved.get("A", 1.3e-11)),
        )
        geometry = DiskGeometry(
            R=_coerce_scalar(resolved["R"]),
            L=_coerce_scalar(resolved["L"]),
        )
        resolved["omega0"] = float(omega0_novosad(material, geometry))
        sources["omega0"] = "computed:omega0_novosad"

    if model_kind == "cpp" and "N" not in resolved:
        resolved["N"] = 0.25
        sources["N"] = "default"
    if model_kind == "cpp" and "domega0_dJ" not in resolved:
        resolved["domega0_dJ"] = 0.0
        sources["domega0_dJ"] = "default"
    if "field" not in resolved:
        resolved["field"] = None
        sources["field"] = "default"

    return AnalyticalParameterResolution(
        resolved_params=resolved,
        param_sources=sources,
        model_kind=model_kind,
        search_locations=search_locations,
    )


__all__ = ["AnalyticalParameterResolution", "extract_model_defaults"]
