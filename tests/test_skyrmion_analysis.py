from __future__ import annotations

import re

import numpy as np
import pytest
import zarr

import mmpp
from mmpp.batch_operations import BatchOperations
from mmpp.core.job import ZarrJobResult
from mmpp.solitons import XYConvention
from mmpp.solitons._method_helpers import CallableNodeHelper
from mmpp.solitons.skyrmion import (
    BatchSkyrmionInterface,
    SizeFitConfig,
    SkyrmionAnalysisResult,
    SkyrmionInterface,
    SkyrmionSizeResult,
    SkyrmionTopologyConfig,
    SkyrmionTopologyResult,
    detect_skyrmion,
    fit_skyrmion_size,
)
from tests.fixtures.synthetic_skyrmion import generate_synthetic_skyrmion

DX = 1e-9
DY = 1e-9


def _create_job(
    tmp_path,
    name: str,
    data: np.ndarray,
    *,
    attrs: dict[str, object] | None = None,
) -> ZarrJobResult:
    zarr_path = tmp_path / f"{name}.zarr"
    group = zarr.open(str(zarr_path), mode="w")
    group.create_dataset("m", data=data, chunks=data.shape)
    group.attrs["dx"] = DX
    group.attrs["dy"] = DY
    group.attrs["t_sampl"] = 1e-12
    for key, value in (attrs or {}).items():
        group.attrs[key] = value
    return ZarrJobResult(str(zarr_path), {})


@pytest.mark.parametrize("y_axis", ["up", "down"])
def test_topology_estimators_have_consistent_charge_sign(y_axis: str):
    field = generate_synthetic_skyrmion(Nx=80, Ny=80, radius=18e-9)
    convention = XYConvention(y_axis=y_axis)

    berg = detect_skyrmion(
        field,
        DX,
        DY,
        convention=convention,
        config=SkyrmionTopologyConfig(method="berg_luscher", convention=convention),
    )
    finite = detect_skyrmion(
        field,
        DX,
        DY,
        convention=convention,
        config=SkyrmionTopologyConfig(method="finite_diff", convention=convention),
    )

    assert berg.state == finite.state == "skyrmion"
    assert berg.valid and finite.valid
    assert abs(berg.Q) > 0.98
    assert abs(finite.Q) > 0.92
    assert np.sign(berg.Q) == np.sign(finite.Q)


def test_topology_detects_reversed_background_and_offset_center():
    center_px = (37.0, 31.0)
    field = generate_synthetic_skyrmion(
        Nx=80,
        Ny=72,
        radius=17e-9,
        center=center_px,
        background_polarity=-1,
    )

    result = detect_skyrmion(field, DX, DY)

    assert result.state == "skyrmion"
    assert result.polarity == 1
    assert result.background_sign == -1
    assert result.center_xy_m[0] == pytest.approx(center_px[0] * DX, abs=1.5 * DX)
    expected_y = (field.shape[0] - 1 - center_px[1]) * DY
    assert result.center_xy_m[1] == pytest.approx(expected_y, abs=1.5 * DY)


def test_ansatz_fit_recovers_radius_and_scale():
    radius = 22e-9
    scale = 3.5e-9
    field = generate_synthetic_skyrmion(
        Nx=96,
        Ny=96,
        radius=radius,
        wall_scale=scale,
        noise=0.002,
    )

    result = fit_skyrmion_size(field, DX, DY, method="ansatz")

    assert result.fit_success
    assert result.model == "ansatz"
    assert result.radius_m == pytest.approx(radius, abs=1.0e-9)
    assert result.scale_m == pytest.approx(scale, rel=0.18)
    assert result.normalized_rmse < 0.02
    assert result.radius_50_m == result.radius_m


def test_auto_fit_selects_gaussian_and_recovers_sigma():
    sigma = 16e-9
    field = generate_synthetic_skyrmion(
        Nx=96,
        Ny=96,
        model="gaussian",
        sigma=sigma,
    )

    result = fit_skyrmion_size(field, DX, DY, method="auto")

    assert result.fit_success
    assert result.model == "gaussian"
    assert result.sigma_m == pytest.approx(sigma, rel=0.05)
    assert result.radius_m == pytest.approx(
        sigma * np.sqrt(2.0 * np.log(2.0)), rel=0.05
    )
    assert result.gaussian_fwhm_m == pytest.approx(
        2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma,
        rel=0.05,
    )
    assert set(result.candidate_diagnostics) == {
        "ansatz",
        "domain_wall",
        "gaussian",
    }


def test_size_config_method_controls_default_auto_argument():
    field = generate_synthetic_skyrmion(Nx=80, Ny=80, radius=18e-9)

    result = fit_skyrmion_size(
        field,
        DX,
        DY,
        config=SizeFitConfig(method="domain_wall"),
    )

    assert result.requested_method == "domain_wall"
    assert result.model == "domain_wall"


def test_zero_filled_geometry_is_excluded_from_topology_stencil():
    field = generate_synthetic_skyrmion(Nx=88, Ny=88, radius=18e-9)
    material = np.zeros(field.shape[:2], dtype=bool)
    material[8:-8, 8:-8] = True
    field[~material] = 0.0

    result = detect_skyrmion(
        field,
        DX,
        DY,
        config=SkyrmionTopologyConfig(method="finite_diff"),
    )

    assert result.state == "skyrmion"
    assert abs(result.Q) > 0.92
    assert np.all(result.q_density[~material] == 0.0)
    assert result.metadata["valid_fraction"] == pytest.approx(np.mean(material))


def test_uniform_field_returns_invalid_reason_coded_results():
    field = np.zeros((64, 64, 3), dtype=float)
    field[..., 2] = 1.0

    topology = detect_skyrmion(field, DX, DY)
    size = fit_skyrmion_size(field, DX, DY, method="threshold")

    assert topology.state == "uniform"
    assert not topology.valid
    assert "insufficient_contrast" in topology.flags
    assert "q_below_threshold" in topology.flags
    assert not size.fit_success
    assert size.quality == "invalid"
    assert "insufficient_contrast" in size.flags


def test_job_dataset_slice_and_shortcut_interfaces(tmp_path):
    first = generate_synthetic_skyrmion(
        Nx=72, Ny=72, radius=15e-9, background_polarity=1
    )
    second = generate_synthetic_skyrmion(
        Nx=72, Ny=72, radius=19e-9, background_polarity=-1
    )
    data = np.stack([first, second], axis=0)[:, np.newaxis, ...]
    job = _create_job(tmp_path, "skyrmion_interfaces", data)

    combined = job.solitons.skyrmion.analyze(frame=1)
    generic_size = job.solitons.skyrmion.analyze("size", frame=1, method="threshold")
    generic_charge = job.solitons.skyrmion.analyze("charge", frame=1)
    direct = job.skyrmion.detect(frame=1)
    dataset = job.m.skyrmion.fit_size(frame=1)
    sliced = job.m[1].skyrmion.fit_size()

    assert isinstance(job.skyrmion, SkyrmionInterface)
    assert isinstance(combined, SkyrmionAnalysisResult)
    assert isinstance(generic_size, SkyrmionSizeResult)
    assert isinstance(generic_charge, SkyrmionTopologyResult)
    assert generic_size.radius_m == pytest.approx(dataset.radius_m)
    assert generic_charge.Q == pytest.approx(direct.Q)
    assert list(job.skyrmion.available_analyses()["analysis"]) == ["size", "charge"]
    assert direct.background_sign == -1
    assert combined.topology.background_sign == -1
    assert dataset.radius_m == pytest.approx(19e-9, abs=1.2e-9)
    assert sliced.radius_m == pytest.approx(19e-9, abs=1.2e-9)


def test_batch_interface_preserves_measurements_and_errors(tmp_path):
    field = generate_synthetic_skyrmion(Nx=72, Ny=72, radius=16e-9)
    good = _create_job(tmp_path, "batch_good", field[np.newaxis, ...])
    bad_data = np.zeros((1, 32, 32, 2), dtype=float)
    bad = _create_job(tmp_path, "batch_bad", bad_data)

    batch = BatchOperations([good, bad], None)
    assert isinstance(batch.skyrmion, BatchSkyrmionInterface)
    assert batch.skyrmion is not batch.solitons.skyrmion
    table = batch.solitons.skyrmion.measure_size(show_progress=False)

    assert list(table["status"]) == ["ok", "error"]
    assert table.loc[0, "radius_nm"] == pytest.approx(16.0, abs=1.2)
    assert table.loc[0, "state"] == "skyrmion"
    assert isinstance(table.loc[1, "error"], str)


def test_batch_size_curve_uses_manual_parameter_and_sorts_results(tmp_path):
    large = generate_synthetic_skyrmion(Nx=72, Ny=72, radius=19e-9)
    small = generate_synthetic_skyrmion(Nx=72, Ny=72, radius=14e-9)
    high_dmi = _create_job(
        tmp_path,
        "sweep_high_dmi",
        large[np.newaxis, ...],
        attrs={"Dind": 3.0e-3},
    )
    low_dmi = _create_job(
        tmp_path,
        "sweep_low_dmi",
        small[np.newaxis, ...],
        attrs={"Dind": 1.5e-3},
    )

    curve = BatchOperations([high_dmi, low_dmi], None).skyrmion.size_vs_parameter(
        "Dind",
        parameter_scale=1e3,
        parameter_unit="mJ/m²",
        method="threshold",
    )

    assert list(curve["parameter_value"]) == [1.5, 3.0]
    assert list(curve["Dind"]) == [1.5, 3.0]
    assert list(curve["parameter_available"]) == [True, True]
    assert curve.iloc[0]["radius_nm"] == pytest.approx(14.0, abs=1.2)
    assert curve.iloc[1]["radius_nm"] == pytest.approx(19.0, abs=1.2)
    assert curve.attrs["parameter"] == "Dind"
    assert curve.attrs["parameter_source"] == "manual"
    assert curve.attrs["parameter_label"] == "Dind [mJ/m²]"


def test_folder_batch_auto_detects_varying_parameter(tmp_path):
    first = generate_synthetic_skyrmion(Nx=64, Ny=64, radius=13e-9)
    second = generate_synthetic_skyrmion(Nx=64, Ny=64, radius=17e-9)
    _create_job(
        tmp_path,
        "folder_sweep_1",
        first[np.newaxis, ...],
        attrs={"B_ext": 0.03, "alpha": 0.01},
    )
    _create_job(
        tmp_path,
        "folder_sweep_2",
        second[np.newaxis, ...],
        attrs={"B_ext": 0.01, "alpha": 0.01},
    )

    jobs = mmpp.open(str(tmp_path), force=True, max_workers=1)
    candidates = jobs[:].skyrmion.parameter_candidates()
    curve = jobs[:].solitons.skyrmion.analyze(
        "size",
        method="threshold",
        parameter_scale=1e3,
        parameter_unit="mT",
    )
    charge_curve = jobs[:].solitons.skyrmion.analyze(
        "charge",
        method="finite_diff",
        parameter_scale=1e3,
        parameter_unit="mT",
    )

    assert len(jobs) == 2
    assert candidates.attrs["recommended_parameter"] == "B_ext"
    assert list(candidates["parameter"]) == ["B_ext"]
    assert list(curve["parameter_value"]) == [10.0, 30.0]
    assert curve.attrs["parameter"] == "B_ext"
    assert curve.attrs["parameter_source"] == "auto"
    assert curve.attrs["parameter_candidates"] == ("B_ext",)
    assert list(curve["analysis"].unique()) == ["size"]
    assert list(curve["observable_name"].unique()) == ["diameter_nm"]
    assert list(curve["observable_value"]) == list(curve["diameter_nm"])
    assert curve.attrs["observable_unit"] == "nm"

    assert list(charge_curve["parameter_value"]) == [10.0, 30.0]
    assert list(charge_curve["analysis"].unique()) == ["charge"]
    assert list(charge_curve["observable_name"].unique()) == ["Q"]
    assert list(charge_curve["observable_value"]) == list(charge_curve["Q"])
    assert charge_curve.attrs["observable_unit"] == "1"


def test_generic_batch_analysis_registry_and_validation(tmp_path):
    field = generate_synthetic_skyrmion(Nx=64, Ny=64, radius=14e-9)
    first = _create_job(
        tmp_path,
        "generic_1",
        field[np.newaxis, ...],
        attrs={"Dind": 1.0e-3},
    )
    second = _create_job(
        tmp_path,
        "generic_2",
        field[np.newaxis, ...],
        attrs={"Dind": 2.0e-3},
    )
    batch = BatchOperations([first, second], None).skyrmion

    available = batch.available_analyses()
    assert list(available["analysis"]) == ["size", "charge"]
    assert "topological_charge" in available.loc[1, "aliases"]

    radius = batch.analyze(
        "size",
        parameter="Dind",
        size_metric="radius_nm",
        method="threshold",
    )
    assert radius.attrs["observable_name"] == "radius_nm"
    assert radius.attrs["observable_unit"] == "nm"
    assert list(radius["observable_value"]) == list(radius["radius_nm"])

    legacy = batch.analyze(method="threshold")
    assert "parameter_value" not in legacy
    assert "radius_nm" in legacy

    with pytest.raises(ValueError, match="Unknown skyrmion analysis"):
        batch.analyze("energy", parameter="Dind")
    with pytest.raises(ValueError, match="Unknown size_metric"):
        batch.analyze("size", parameter="Dind", size_metric="area_nm2")
    with pytest.raises(ValueError, match="requires an observable"):
        batch.analyze(parameter="Dind", method="threshold")


def test_batch_parameter_auto_detection_reports_missing_variation(tmp_path):
    field = generate_synthetic_skyrmion(Nx=64, Ny=64, radius=14e-9)
    first = _create_job(
        tmp_path,
        "constant_1",
        field[np.newaxis, ...],
        attrs={"B_ext": 0.02},
    )
    second = _create_job(
        tmp_path,
        "constant_2",
        field[np.newaxis, ...],
        attrs={"B_ext": 0.02},
    )
    batch = BatchOperations([first, second], None).skyrmion

    assert batch.parameter_candidates().empty
    with pytest.raises(ValueError, match="Could not auto-detect"):
        batch.size_vs_parameter(method="threshold")
    with pytest.raises(ValueError, match="does not vary"):
        batch.size_vs_parameter(parameter="B_ext", method="threshold")


def test_skyrmion_html_helpers_use_canonical_cards(tmp_path):
    field = generate_synthetic_skyrmion(Nx=64, Ny=64, radius=14e-9)
    job = _create_job(tmp_path, "skyrmion_html", field[np.newaxis, ...])
    namespace = job.skyrmion
    batch_namespace = BatchOperations([job], None).solitons
    topology = namespace.detect()
    size = namespace.fit_size()

    callable_nodes = [
        namespace.detect,
        namespace.measure_size,
        namespace.fit_size,
        namespace.available_analyses,
        namespace.analyze,
        namespace.interactive,
        namespace.interactive_spectrum,
        namespace.interactive_modes,
        namespace.topology.detect,
        namespace.topology.topological_charge,
        namespace.topology.center,
        namespace.size.fit,
        namespace.size.measure,
        BatchSkyrmionInterface([job]).parameter_candidates,
        BatchSkyrmionInterface([job]).available_analyses,
        BatchSkyrmionInterface([job]).detect,
        BatchSkyrmionInterface([job]).measure_size,
        BatchSkyrmionInterface([job]).size_vs_parameter,
        BatchSkyrmionInterface([job]).fit_size,
        BatchSkyrmionInterface([job]).analyze,
        BatchSkyrmionInterface([job]).interactive,
    ]
    for node in callable_nodes:
        assert isinstance(node, CallableNodeHelper)
        assert callable(node)
        helper_html = node._repr_html_()
        assert ">Overview</button>" in helper_html
        assert ">API</button>" in helper_html
        assert "box-shadow" in helper_html
        assert "<h3" not in helper_html

    cards = [
        job.solitons._repr_html_(),
        namespace._repr_html_(),
        namespace.topology._repr_html_(),
        namespace.size._repr_html_(),
        topology._repr_html_(),
        size._repr_html_(),
        batch_namespace._repr_html_(),
        BatchSkyrmionInterface([job])._repr_html_(),
    ]

    for card in cards:
        assert "API" in card
        assert "<h3" not in card.lower()
        assert "box-shadow" in card
        assert "onclick=" in card
        assert "-tab-0" in card
        assert "-panel-0" in card

    assert "SkyrmionTopologyResult" in topology._repr_html_()
    assert "SkyrmionSizeResult" in size._repr_html_()
    assert "SkyrmionAnalysisResult" in namespace.analyze()._repr_html_()
    assert "size_vs_parameter" in cards[-1]
    assert "parameter_candidates" in cards[-1]
    assert "available_analyses" in cards[-1]
    assert "analyze(&#x27;size&#x27;)" in cards[-1] or "analyze('size')" in cards[-1]
    assert (
        "analyze(&#x27;charge&#x27;)" in cards[-1] or "analyze('charge')" in cards[-1]
    )

    solitons_card = cards[0]
    assert "Skyrmion Workflow" in solitons_card
    assert "skyrmion.fit_size" in solitons_card
    assert "skyrmion.analyze" in solitons_card

    first_id = re.search(r"id='([^']+-tab-0)'", namespace._repr_html_())
    second_id = re.search(r"id='([^']+-tab-0)'", namespace._repr_html_())
    assert first_id is not None and second_id is not None
    assert first_id.group(1) != second_id.group(1)

    first_batch_id = re.search(r"id='([^']+-tab-0)'", batch_namespace._repr_html_())
    second_batch_id = re.search(r"id='([^']+-tab-0)'", batch_namespace._repr_html_())
    assert first_batch_id is not None and second_batch_id is not None
    assert first_batch_id.group(1) != second_batch_id.group(1)


def test_skyrmion_interactive_dashboard_renders_three_panel_result(tmp_path):
    pytest.importorskip("ipywidgets")
    field = generate_synthetic_skyrmion(Nx=64, Ny=64, radius=14e-9)
    job = _create_job(tmp_path, "skyrmion_interactive", field[np.newaxis, ...])

    dashboard = job.skyrmion.interactive(
        show=False,
        initial_frame=0,
        z_layer=-1,
        topology_method="berg_luscher",
        size_method="threshold",
    )
    topology, size = dashboard.run()

    assert topology.valid
    assert size.radius_nm == pytest.approx(14.0, abs=1.2)
    assert dashboard.last_topology is topology
    assert dashboard.last_size is size
    assert len(dashboard.image.value) > 1000
    assert "Done" in dashboard.status.value
    assert "interactive" in job.skyrmion._repr_html_()


def test_skyrmion_spectrum_and_modes_preserve_dataset_slice():
    calls = []
    explorer = object()

    class _Modes:
        def interactive_spectrum(self, **kwargs):
            calls.append(kwargs)
            return explorer

    class _FFT:
        modes = _Modes()

    class _Data:
        fft = _FFT()

        def __getitem__(self, key):
            calls.append(("slice", key))
            return self

    class _Job:
        m = _Data()

    selection = (slice(0, 10), Ellipsis)
    interface = SkyrmionInterface(_Job(), dataset_name="m", slice_info=selection)

    assert interface.interactive_spectrum(dpi=140) is explorer
    assert interface.interactive_modes(dpi=160) is explorer
    assert calls == [
        ("slice", selection),
        {"dpi": 140},
        ("slice", selection),
        {"dpi": 160},
    ]


def test_soliton_spectral_viewer_keeps_bound_dataset_wrapper(tmp_path):
    from mmpp.solitons._spectral_ui import dataset_view

    field = generate_synthetic_skyrmion(Nx=32, Ny=32, radius=8e-9)
    data = np.stack([field, field], axis=0)
    job = _create_job(tmp_path, "bound_spectral_view", data)
    view = job.m[:1]

    assert dataset_view(view.skyrmion) is view
    assert dataset_view(view.vortex) is view


def test_skyrmion_dashboard_selects_spectrum_and_modes(tmp_path):
    pytest.importorskip("ipywidgets")
    field = generate_synthetic_skyrmion(Nx=48, Ny=48, radius=11e-9)
    job = _create_job(tmp_path, "skyrmion_spectral_dashboard", field[np.newaxis, ...])
    dashboard = job.skyrmion.interactive(show=False, initial_module="spectrum")
    calls = []
    spectrum_viewer = object()
    mode_viewer = object()
    dashboard._interface.interactive_spectrum = lambda **kwargs: (
        calls.append(("spectrum", kwargs)) or spectrum_viewer
    )
    dashboard._interface.interactive_modes = lambda **kwargs: (
        calls.append(("modes", kwargs)) or mode_viewer
    )

    assert dashboard.module.value == "spectrum"
    assert dashboard.run_selected() is spectrum_viewer
    dashboard.module.value = "modes"
    assert dashboard.run_selected() is mode_viewer
    assert dashboard.last_spectral_viewer is mode_viewer
    assert calls == [
        ("spectrum", {"z_layer": -1, "dpi": 110}),
        ("modes", {"z_layer": -1, "dpi": 110}),
    ]


def test_batch_skyrmion_interactive_selects_sorted_result(tmp_path):
    pytest.importorskip("ipywidgets")
    field = generate_synthetic_skyrmion(Nx=48, Ny=48, radius=11e-9)
    high = _create_job(
        tmp_path,
        "interactive_high",
        field[np.newaxis, ...],
        attrs={"Dind": 3.0e-3},
    )
    low = _create_job(
        tmp_path,
        "interactive_low",
        field[np.newaxis, ...],
        attrs={"Dind": 1.0e-3},
    )

    dashboard = BatchOperations([high, low], None).skyrmion.interactive(
        index=0,
        sort_by="Dind",
        show=False,
    )

    assert dashboard._interface._job is low
