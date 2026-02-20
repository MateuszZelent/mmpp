"""Animation callback helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .filters import component_plot_label, resolve_mode_components


def _collect_mode_images_and_titles(explorer: Any) -> tuple[list[list[Any]], list[list[Any]]]:
    """Collect current mode subplot images and title handles."""
    mode_images: list[list[Any]] = []
    mode_titles: list[list[Any]] = []

    if explorer._mode_axes is None:
        return mode_images, mode_titles

    for row_idx, _row_type in enumerate(explorer._mode_row_types):
        row_images: list[Any] = []
        row_titles: list[Any] = []
        for col_idx in range(explorer._mode_axes.shape[1]):
            ax = explorer._mode_axes[row_idx, col_idx]
            img = None
            for child in ax.get_children():
                from matplotlib.image import AxesImage

                if isinstance(child, AxesImage):
                    img = child
                    break
            row_images.append(img)
            row_titles.append(ax.title)

        mode_images.append(row_images)
        mode_titles.append(row_titles)

    return mode_images, mode_titles


def _resolve_mode_viz(
    comp_data: np.ndarray,
    *,
    viz_type: str,
    use_holography: bool = False,
) -> tuple[np.ndarray, float | None, float | None]:
    """Resolve frame data and clim values for a visualization mode."""
    comp_amplitude = float(np.nanmax(np.abs(comp_data)))
    if comp_amplitude <= 0:
        comp_amplitude = 1.0

    if viz_type == "phase" and use_holography:
        try:
            from ..vortex_optics import VortexOptics

            return VortexOptics.complex_holography(comp_data), None, None
        except Exception:
            pass

    if viz_type in {"magnitude", "abs"}:
        return np.abs(comp_data), 0.0, comp_amplitude
    if viz_type == "phase":
        return np.angle(comp_data), -np.pi, np.pi
    if viz_type == "imag":
        return np.imag(comp_data), -comp_amplitude, comp_amplitude
    # "real" and "combined" are rendered as real part in time-domain animation
    return np.real(comp_data), -comp_amplitude, comp_amplitude


def on_save_animation_clicked(explorer: Any, _btn: Any) -> None:
    """Save phase oscillation animation of selected FMR mode."""
    if explorer._is_saving_animation:
        return
    if explorer._fig is None or explorer._current_frequency_ghz is None:
        explorer._set_status("No mode selected to animate", color="crimson")
        return
    if "save_animation" not in explorer._controls:
        return

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        try:
            from matplotlib.animation import FFMpegWriter
        except Exception:  # pragma: no cover - optional backend
            FFMpegWriter = None  # type: ignore[assignment]
        try:
            from matplotlib.animation import ImageMagickWriter
            has_imagemagick = True
        except Exception:  # pragma: no cover - optional backend
            ImageMagickWriter = None  # type: ignore[assignment]
            has_imagemagick = False
    except Exception as exc:  # pragma: no cover - optional backend
        explorer._set_status(f"Animation backend unavailable: {exc}", color="crimson")
        return

    n_frames = max(2, int(explorer._controls["anim_frames"].value))
    fps = max(1, int(explorer._controls["anim_fps"].value))
    fmt = str(explorer._controls["anim_format"].value).lower()
    if fmt not in {"gif", "mp4"}:
        fmt = "gif"

    button = explorer._controls["save_animation"]
    old_desc = button.description

    explorer._is_saving_animation = True
    button.disabled = True
    button.description = "Loading mode..."
    explorer._set_status("Loading mode data...", color="#0F766E")

    try:
        freq_ghz = explorer._current_frequency_ghz
        mode_array, actual_freq, _extent = explorer._load_mode(
            freq_ghz, explorer._current_z_layer
        )

        freq_hz = actual_freq * 1e9
        omega = 2 * np.pi * freq_hz
        period_s = 1.0 / freq_hz
        time_array = np.linspace(0, period_s, n_frames, endpoint=False)

        button.description = "Pre-computing..."
        explorer._set_status(f"Pre-computing {n_frames} frames...", color="#0F766E")

        precomputed_frames = []
        for i, t in enumerate(time_array):
            phase_factor = np.exp(-1j * omega * t)
            mode_at_t = mode_array * phase_factor
            precomputed_frames.append(mode_at_t)

            if i % max(1, n_frames // 10) == 0:
                button.description = f"Frame {i + 1}/{n_frames}"

        button.description = "Rendering..."
        explorer._set_status("Rendering animation...", color="#0F766E")

        mode_images, mode_titles = _collect_mode_images_and_titles(explorer)
        mode_type = str(getattr(explorer, "_mode_type", "combined"))

        def _update_frame(frame_idx: int) -> list[Any]:
            mode_at_t = precomputed_frames[frame_idx]
            t = time_array[frame_idx]
            phase_deg = (t / period_s) * 360
            t_ns = t * 1e9

            artists = []
            resolved_components = resolve_mode_components(
                mode_at_t, explorer._current_components
            )

            for row_idx, row_type in enumerate(explorer._mode_row_types):
                for col_idx, comp in enumerate(explorer._current_components):
                    if row_idx >= len(mode_images) or col_idx >= len(mode_images[row_idx]):
                        continue
                    img = mode_images[row_idx][col_idx]
                    if img is None:
                        continue

                    comp_data = resolved_components.get(comp)
                    if comp_data is None:
                        continue

                    viz_type = mode_type if row_idx == 0 else row_type
                    use_holography = bool(
                        getattr(explorer, "_use_holography", False)
                        and viz_type == "phase"
                    )
                    plot_data, vmin, vmax = _resolve_mode_viz(
                        comp_data,
                        viz_type=viz_type,
                        use_holography=use_holography,
                    )

                    img.set_data(plot_data)
                    if vmin is not None and vmax is not None:
                        img.set_clim(vmin, vmax)
                    artists.append(img)

                    if row_idx == 0 and row_idx < len(mode_titles):
                        title_obj = mode_titles[row_idx][col_idx]
                        title_obj.set_text(
                            f"{component_plot_label(comp)} @ {actual_freq:.3f} GHz | t={t_ns:.2f}ns | φ={phase_deg:.0f}°"
                        )
                        artists.append(title_obj)

            return artists

        animation = FuncAnimation(
            explorer._fig,
            _update_frame,
            frames=n_frames,
            interval=1000.0 / float(fps),
            blit=False,
            repeat=False,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path.cwd() / f"fmr_mode_{actual_freq:.2f}GHz_{mode_type}_{timestamp}.{fmt}"

        writer_name = ""
        export_dpi = 150 if fmt == "gif" else explorer.dpi
        if fmt == "mp4":
            if FFMpegWriter is None:
                raise RuntimeError("FFmpeg writer unavailable; select GIF format")
            writer = FFMpegWriter(
                fps=fps,
                bitrate=4000,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-preset", "slower", "-crf", "18"],
            )
            writer_name = "FFmpeg"
        else:
            if has_imagemagick and ImageMagickWriter is not None:
                try:
                    writer = ImageMagickWriter(
                        fps=fps,
                        metadata={"Author": "MMPP", "Title": "FMR Mode Animation"},
                        bitrate=2000,
                        extra_args=["-layers", "Optimize"],
                    )
                    writer_name = "ImageMagick"
                except Exception:
                    writer = PillowWriter(
                        fps=fps,
                        metadata={"Author": "MMPP", "Title": "FMR Mode Animation"},
                    )
                    writer_name = "Pillow (256 colors)"
            else:
                writer = PillowWriter(
                    fps=fps,
                    metadata={"Author": "MMPP", "Title": "FMR Mode Animation"},
                )
                writer_name = "Pillow (256 colors)"

        explorer._set_status(f"Saving animation: {output_path.name}...", color="#0F766E")
        animation.save(str(output_path), writer=writer, dpi=export_dpi)
        size_mb = output_path.stat().st_size / (1024 * 1024)

        quality_hint = ""
        if fmt == "gif" and writer_name == "Pillow (256 colors)":
            quality_hint = " | tip: use MP4 for better color quality"

        explorer._set_status(
            (
                f"Saved: {output_path.name} ({size_mb:.1f} MB) | "
                f"{n_frames} frames @ {fps} fps, {export_dpi} dpi, writer={writer_name}"
                f"{quality_hint}"
            ),
            color="seagreen",
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        explorer._set_status(f"Animation failed: {exc}", color="crimson")
    finally:
        button.disabled = False
        button.description = old_desc
        explorer._is_saving_animation = False


def on_animate_clicked(explorer: Any, _btn: Any) -> None:
    """Toggle live animation preview of selected mode."""
    if explorer._fig is None or explorer._current_frequency_ghz is None:
        explorer._set_status("No mode selected to animate", color="crimson")
        return

    if explorer._is_animating:
        stop_animation(explorer)
        if "animate" in explorer._controls:
            explorer._controls["animate"].description = "🎬 Animate"
            explorer._controls["animate"].button_style = "warning"
        explorer._set_status("Animation stopped", color="seagreen")
        explorer._update_mode_plots()
        return

    try:
        from matplotlib.animation import FuncAnimation

        freq_ghz = explorer._current_frequency_ghz
        mode_array, actual_freq, _extent = explorer._load_mode(
            freq_ghz, explorer._current_z_layer
        )

        freq_hz = actual_freq * 1e9
        omega = 2 * np.pi * freq_hz
        period_s = 1.0 / freq_hz

        n_frames = int(
            explorer._controls.get("anim_frames", {}).value
            if hasattr(explorer._controls.get("anim_frames"), "value")
            else 60
        )
        fps = int(
            explorer._controls.get("anim_fps", {}).value
            if hasattr(explorer._controls.get("anim_fps"), "value")
            else 24
        )

        time_array = np.linspace(0, period_s, n_frames, endpoint=False)

        precomputed_frames = []
        for t in time_array:
            phase_factor = np.exp(-1j * omega * t)
            mode_at_t = mode_array * phase_factor
            precomputed_frames.append(mode_at_t)

        mode_images, mode_titles = _collect_mode_images_and_titles(explorer)
        mode_type = explorer._mode_type

        def _update_frame(frame_idx: int) -> list[Any]:
            mode_at_t = precomputed_frames[frame_idx]
            t = time_array[frame_idx]
            phase_deg = (t / period_s) * 360
            t_ns = t * 1e9

            artists = []
            resolved_components = resolve_mode_components(
                mode_at_t, explorer._current_components
            )

            for row_idx, row_type in enumerate(explorer._mode_row_types):
                for col_idx, comp in enumerate(explorer._current_components):
                    if row_idx >= len(mode_images) or col_idx >= len(mode_images[row_idx]):
                        continue
                    img = mode_images[row_idx][col_idx]
                    if img is None:
                        continue

                    comp_data = resolved_components.get(comp)
                    if comp_data is None:
                        continue

                    viz_type = mode_type if row_idx == 0 else row_type

                    if viz_type in ["magnitude", "abs"]:
                        plot_data = np.abs(comp_data)
                    elif viz_type == "phase":
                        if getattr(explorer, "_use_holography", False):
                            try:
                                from ..vortex_optics import VortexOptics

                                plot_data = VortexOptics.complex_holography(comp_data)
                            except Exception:
                                plot_data = np.angle(comp_data)
                        else:
                            plot_data = np.angle(comp_data)
                    elif viz_type == "real":
                        plot_data = np.real(comp_data)
                    elif viz_type == "imag":
                        plot_data = np.imag(comp_data)
                    else:
                        plot_data = np.real(comp_data)

                    img.set_data(plot_data)
                    artists.append(img)

                    if row_idx == 0 and row_idx < len(mode_titles):
                        title_obj = mode_titles[row_idx][col_idx]
                        title_obj.set_text(
                            f"{component_plot_label(comp)} @ {actual_freq:.3f} GHz | t={t_ns:.2f}ns | φ={phase_deg:.0f}°"
                        )
                        artists.append(title_obj)

            return artists

        explorer._animation = FuncAnimation(
            explorer._fig,
            _update_frame,
            frames=n_frames,
            interval=1000.0 / float(fps),
            blit=True,
            repeat=True,
        )

        explorer._is_animating = True
        if "animate" in explorer._controls:
            explorer._controls["animate"].description = "⏸️ Stop"
            explorer._controls["animate"].button_style = "danger"

        explorer._set_status(
            f"Animating: {n_frames} frames, T={period_s*1e9:.2f}ns (1 period)",
            color="seagreen",
        )

        explorer._fig.canvas.draw_idle()

    except Exception as exc:
        import traceback

        traceback.print_exc()
        explorer._set_status(f"Animation error: {exc}", color="crimson")
        explorer._is_animating = False
        if "animate" in explorer._controls:
            explorer._controls["animate"].description = "🎬 Animate"
            explorer._controls["animate"].button_style = "warning"


def stop_animation(explorer: Any) -> None:
    """Stop any running animation."""
    if explorer._animation is not None:
        try:
            explorer._animation.event_source.stop()
        except Exception:
            pass
        explorer._animation = None
    explorer._is_animating = False


def on_mode_type_changed(explorer: Any, change: Any) -> None:
    """Handle mode visualization type change."""
    if explorer._internal_update:
        return
    new_type = change.get("new", "combined")
    explorer._mode_type = new_type

    if explorer._is_animating:
        stop_animation(explorer)
        on_animate_clicked(explorer, None)
    else:
        explorer._update_mode_plots()


def on_phase_index_changed(explorer: Any, change: Any) -> None:
    """Handle phase slider updates for static phase preview."""
    if explorer._internal_update:
        return
    if explorer._fig is None or explorer._current_frequency_ghz is None:
        return
    if explorer._mode_axes is None:
        return

    phase_idx = change.get("new", 0)
    n_frames = 60

    phase_rad = (phase_idx / n_frames) * 2 * np.pi
    phase_deg = (phase_idx / n_frames) * 360

    try:
        mode_array, actual_freq, _extent = explorer._load_mode(
            explorer._current_frequency_ghz,
            explorer._current_z_layer,
        )

        phase_factor = np.exp(-1j * phase_rad)
        mode_at_phase = mode_array * phase_factor
        mode_type = explorer._mode_type
        resolved_components = resolve_mode_components(
            mode_at_phase, explorer._current_components
        )

        for row_idx, row_type in enumerate(explorer._mode_row_types):
            for col_idx, comp in enumerate(explorer._current_components):
                if (
                    row_idx >= explorer._mode_axes.shape[0]
                    or col_idx >= explorer._mode_axes.shape[1]
                ):
                    continue

                ax = explorer._mode_axes[row_idx, col_idx]
                xlim_saved = ax.get_xlim()
                ylim_saved = ax.get_ylim()

                img = None
                for child in ax.get_children():
                    from matplotlib.image import AxesImage

                    if isinstance(child, AxesImage):
                        img = child
                        break

                if img is None:
                    continue

                comp_data = resolved_components.get(comp)
                if comp_data is None:
                    continue

                comp_amplitude = float(np.nanmax(np.abs(comp_data)))
                if comp_amplitude <= 0:
                    comp_amplitude = 1.0

                if row_type == "magnitude" or mode_type == "abs":
                    plot_data = np.abs(comp_data)
                    img.set_clim(0, comp_amplitude)
                elif row_type == "phase":
                    if getattr(explorer, "_use_holography", False):
                        try:
                            from ..vortex_optics import VortexOptics

                            plot_data = VortexOptics.complex_holography(comp_data)
                        except Exception:
                            plot_data = np.angle(comp_data)
                            img.set_clim(-np.pi, np.pi)
                    else:
                        plot_data = np.angle(comp_data)
                        img.set_clim(-np.pi, np.pi)
                elif mode_type == "real" or row_type == "combined":
                    plot_data = np.real(comp_data)
                    img.set_clim(-comp_amplitude, comp_amplitude)
                elif mode_type == "imag":
                    plot_data = np.imag(comp_data)
                    img.set_clim(-comp_amplitude, comp_amplitude)
                else:
                    plot_data = np.real(comp_data)
                    img.set_clim(-comp_amplitude, comp_amplitude)

                img.set_data(plot_data)

                ax.set_xlim(xlim_saved)
                ax.set_ylim(ylim_saved)

                if row_idx == 0:
                    freq_hz = actual_freq * 1e9
                    t_ns = (phase_idx / n_frames) * (1.0 / freq_hz) * 1e9
                    ax.set_title(
                        f"{component_plot_label(comp)} @ {actual_freq:.3f} GHz | t={t_ns:.2f}ns | φ={phase_deg:.0f}°",
                        fontsize=10,
                    )

        if explorer._fig is not None:
            explorer._fig.canvas.draw_idle()

    except Exception:
        import traceback

        traceback.print_exc()


__all__ = [
    "on_save_animation_clicked",
    "on_animate_clicked",
    "stop_animation",
    "on_mode_type_changed",
    "on_phase_index_changed",
]
