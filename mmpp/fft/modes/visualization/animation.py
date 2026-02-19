"""
Animation functionality for FMR modes visualization.

This module contains all animation-related functions:
- save_modes_animation: Save temporal/frequency/phase animations
- Mode animation in interactive plots (toggle, start, stop)
- FFmpeg integration and fallback support
"""

import numpy as np
import logging
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import FMRModeData

log = logging.getLogger("mmpp.fft.modes")

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
    ANIMATION_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    ANIMATION_AVAILABLE = False

# Import FFmpeg utilities
from ..ffmpeg_utils import _ensure_ffmpeg_available, _create_ffmpeg_writer
from ..style import setup_animation_styling, MidpointNormalize


def save_modes_animation(
    analyzer,
    frequency_range: tuple[float, float] = None,
    frequency: float = None,
    save_path: str = "mode_animation.gif",
    fps: int = 15,
    z_layer: int = 0,
    component: Union[str, int] = "z",
    animation_type: str = "temporal",
    colormap: str = None,
    use_midpoint_norm: bool = None,
    figsize: tuple[float, float] = None,
) -> None:
    """
    Save animation of FMR modes.

    Parameters:
    -----------
    analyzer : FMRModeAnalyzer
        The analyzer instance
    frequency_range : tuple, optional
        (f_min, f_max) in GHz for frequency sweep animation
    frequency : float, optional
        Single frequency for temporal animation (in GHz)
    save_path : str
        Output file path (.gif or .mp4)
    fps : int
        Frames per second (default: 15)
    z_layer : int
        Z-layer to animate (default: 0)
    component : str or int
        Component to animate (default: 'z')
    animation_type : str
        Type of animation:
        - 'temporal': Real part of mode oscillating in time at fixed frequency
        - 'frequency': Mode amplitude across frequency range
        - 'phase': Phase evolution at fixed frequency
    colormap : str, optional
        Colormap name. If None, uses config defaults
    use_midpoint_norm : bool, optional
        Use symmetric normalization around zero (default: from config)
    figsize : tuple, optional
        Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE or not ANIMATION_AVAILABLE:
        raise ImportError(
            "Matplotlib and animation support are required for animations"
        )

    # Setup professional styling for animations
    setup_animation_styling(use_paper_style=True, use_custom_fonts=True)

    try:
        import matplotlib.cm as cm

        # Parameter validation
        if frequency_range is None and frequency is None:
            raise ValueError(
                "Either frequency_range or frequency must be specified"
            )

        if frequency_range is not None and frequency is not None:
            raise ValueError(
                "Specify either frequency_range OR frequency, not both"
            )

        # Set defaults with intelligent choices for animation type
        figsize = figsize or (10, 8)
        colormap = colormap or analyzer.config.colormap_animation

        # Auto-enable MidpointNormalize for temporal animations if not explicitly set
        if use_midpoint_norm is None:
            if animation_type == "temporal":
                use_midpoint_norm = True  # Temporal animations benefit from symmetric normalization
            else:
                use_midpoint_norm = analyzer.config.use_midpoint_norm

        # Choose better colormap for temporal oscillations if default 'balance'
        if animation_type == "temporal" and colormap == "balance":
            log.info(
                "Using diverging colormap 'balance' - perfect for oscillating modes"
            )

        # Try to get cmocean colormaps for better scientific visualization
        try:
            import cmocean

            if colormap == "balance":
                cmap = cmocean.cm.balance  # Perfect for data with +/- symmetry
            elif colormap == "diff":
                cmap = cmocean.cm.diff  # Another good diverging colormap
            elif colormap == "curl":
                cmap = cmocean.cm.curl  # Good for circular/phase data
            elif colormap == "delta":
                cmap = cmocean.cm.delta  # Good for deviations from mean
            elif colormap == "tarn":
                cmap = cmocean.cm.tarn  # Good for complex data
            else:
                # Try as regular matplotlib colormap
                cmap = plt.get_cmap(colormap)
        except ImportError:
            log.warning("cmocean not available, using matplotlib colormaps")
            if colormap == "balance":
                cmap = plt.get_cmap("RdBu_r")  # Best fallback for balance
            elif colormap == "diff":
                cmap = plt.get_cmap("RdBu")  # Alternative diverging
            elif colormap == "curl" or colormap == "tarn":
                cmap = plt.get_cmap("RdYlBu_r")  # Complex fallback
            elif colormap == "delta":
                cmap = plt.get_cmap("PuOr_r")  # Another diverging option
            else:
                cmap = plt.get_cmap(colormap)

        # Setup figure
        fig, ax = plt.subplots(figsize=figsize, dpi=analyzer.config.dpi)

        if animation_type == "temporal" and frequency is not None:
            # Temporal animation: Real part oscillating in time (true physical dynamics)
            log.info(f"Creating temporal animation at {frequency:.3f} GHz")

            # Get mode data
            mode_data = analyzer.get_mode(frequency, z_layer)
            comp_data = mode_data.get_component(component)

            # Get amplitude and phase - this is the complex mode from FFT
            amplitude = np.abs(comp_data)
            phase = np.angle(comp_data)

            # Setup normalization - MidpointNormalize is perfect for oscillating data
            if use_midpoint_norm:
                vmax = np.max(amplitude)
                norm = MidpointNormalize(vmin=-vmax, vmax=vmax, midpoint=0)
            else:
                vmax = np.max(amplitude)
                norm = plt.Normalize(vmin=-vmax, vmax=vmax)

            # Time steps for one full oscillation period
            time_steps = np.linspace(0, 2 * np.pi, analyzer.config.animation_time_steps)

            # Create initial plot for colorbar setup
            t = time_steps[0]
            real_part = amplitude * np.cos(phase + t)
            im = ax.imshow(
                real_part,
                cmap=cmap,
                norm=norm,
                extent=mode_data.extent,
                aspect="equal",
                origin="lower",
                interpolation=analyzer.config.interpolation,
            )

            # Create colorbar once
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Magnetization (arb. units)")

            # Set labels once (they won't change)
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")

            def animate_temporal(frame):
                # Calculate real part at this time step
                t = time_steps[frame]
                real_part = amplitude * np.cos(phase + t)

                # Update image data instead of recreating
                im.set_array(real_part)

                # Show fraction of period completed
                period_fraction = t / (2 * np.pi)
                ax.set_title(
                    f"Re[m_{component}] @ {frequency:.3f} GHz (t = {period_fraction:.2f}T)"
                )

                return [im]

            # Create animation
            anim = FuncAnimation(
                fig,
                animate_temporal,
                frames=len(time_steps),
                interval=1000 / fps,
                blit=True,
                repeat=True,
            )

        elif animation_type == "frequency" and frequency_range is not None:
            # Frequency sweep animation
            f_min, f_max = frequency_range
            freq_mask = (analyzer.frequencies >= f_min) & (analyzer.frequencies <= f_max)
            freq_indices = np.where(freq_mask)[0]

            if len(freq_indices) == 0:
                raise ValueError("No frequencies found in specified range")

            log.info(
                f"Creating frequency sweep animation: {f_min:.3f} - {f_max:.3f} GHz"
            )

            # Pre-calculate all mode data for consistent normalization
            all_amplitudes = []
            for freq_idx in freq_indices:
                freq = analyzer.frequencies[freq_idx]
                mode_data = analyzer.get_mode(freq, z_layer)
                comp_data = mode_data.get_component(component)
                all_amplitudes.append(np.abs(comp_data))

            # Global normalization
            global_max = np.max([np.max(amp) for amp in all_amplitudes])
            norm = plt.Normalize(vmin=0, vmax=global_max)

            # Create initial plot for colorbar setup
            mode_data = analyzer.get_mode(analyzer.frequencies[freq_indices[0]], z_layer)
            im = ax.imshow(
                all_amplitudes[0],
                cmap=cmap,
                norm=norm,
                extent=mode_data.extent,
                aspect="equal",
                origin="lower",
                interpolation=analyzer.config.interpolation,
            )

            # Create colorbar once
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("|Magnetization| (arb. units)")

            # Set labels once (they won't change)
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")

            def animate_frequency(frame):
                freq_idx = freq_indices[frame]
                frequency = analyzer.frequencies[freq_idx]
                amplitude = all_amplitudes[frame]

                # Update image data instead of recreating
                im.set_array(amplitude)

                ax.set_title(f"|m_{component}| @ {frequency:.3f} GHz")

                return [im]

            # Create animation
            anim = FuncAnimation(
                fig,
                animate_frequency,
                frames=len(freq_indices),
                interval=1000 / fps,
                blit=True,
                repeat=True,
            )

        elif animation_type == "phase" and frequency is not None:
            # Phase evolution animation
            log.info(f"Creating phase animation at {frequency:.3f} GHz")

            mode_data = analyzer.get_mode(frequency, z_layer)
            comp_data = mode_data.get_component(component)

            amplitude = np.abs(comp_data)
            phase = np.angle(comp_data)

            # Phase steps
            phase_steps = np.linspace(
                0, 2 * np.pi, analyzer.config.animation_time_steps
            )

            # Create initial plot for colorbar setup
            current_phase = (phase + phase_steps[0]) % (2 * np.pi)
            im = ax.imshow(
                current_phase,
                cmap="hsv",  # HSV is perfect for phase
                vmin=0,
                vmax=2 * np.pi,
                extent=mode_data.extent,
                aspect="equal",
                origin="lower",
                interpolation=analyzer.config.interpolation,
            )

            # Create colorbar once
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Phase (rad)")
            cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])

            # Set labels once (they won't change)
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")

            def animate_phase(frame):
                # Add phase offset
                current_phase = (phase + phase_steps[frame]) % (2 * np.pi)

                # Update image data instead of recreating
                im.set_array(current_phase)

                ax.set_title(
                    f"Phase[m_{component}] @ {frequency:.3f} GHz (φ offset = {phase_steps[frame]:.2f})"
                )

                return [im]

            # Create animation
            anim = FuncAnimation(
                fig,
                animate_phase,
                frames=len(phase_steps),
                interval=1000 / fps,
                blit=True,
                repeat=True,
            )

        else:
            raise ValueError(
                f"Invalid animation_type '{animation_type}' for given parameters"
            )

        # Save animation
        plt.tight_layout()

        # Choose writer based on file extension with fallback support
        if save_path.endswith(".mp4"):
            # Ensure FFmpeg is available with auto-installation
            ffmpeg_path = _ensure_ffmpeg_available()
            if ffmpeg_path:
                writer = "ffmpeg"
                log.info("Using FFmpeg writer for MP4 format")
            else:
                log.warning(
                    "FFmpeg not available on system, converting to GIF format"
                )
                save_path = save_path.replace(".mp4", ".gif")
                writer = "pillow"
        elif save_path.endswith(".gif"):
            writer = "pillow"
        else:
            writer = "pillow"  # Default to GIF
            if not save_path.endswith(".gif"):
                save_path += ".gif"

        log.info(f"Saving animation to {save_path} (this may take a while...)")

        try:
            # Save animation with error handling
            if writer == "ffmpeg" and save_path.endswith(".mp4"):
                # Try to save with ffmpeg writer
                try:
                    anim.save(
                        save_path,
                        writer=writer,
                        fps=fps,
                        dpi=analyzer.config.dpi // 2,
                    )
                except Exception as ffmpeg_error:
                    # If FFmpeg fails, try with a more basic writer configuration
                    log.warning(
                        f"FFmpeg save with advanced options failed: {ffmpeg_error}"
                    )
                    log.info("Trying basic FFmpeg configuration...")
                    anim.save(
                        save_path,
                        writer="ffmpeg",
                        fps=fps,
                        dpi=analyzer.config.dpi // 2,
                    )
            else:
                anim.save(
                    save_path, writer=writer, fps=fps, dpi=analyzer.config.dpi // 2
                )

            log.info("✅ Animation saved successfully!")

        except Exception as save_error:
            log.error(f"Failed to save animation with {writer}: {save_error}")

            # Fallback to GIF if MP4 fails
            if writer == "ffmpeg" and save_path.endswith(".mp4"):
                log.info("Attempting fallback to GIF format...")
                fallback_path = save_path.replace(".mp4", ".gif")
                try:
                    anim.save(
                        fallback_path,
                        writer="pillow",
                        fps=fps,
                        dpi=analyzer.config.dpi // 2,
                    )
                    log.info(f"✅ Animation saved as GIF: {fallback_path}")
                except Exception as gif_error:
                    log.error(f"Fallback to GIF also failed: {gif_error}")
                    raise save_error
            else:
                raise save_error

        plt.close(fig)

    except ImportError as e:
        log.error(f"Animation requires additional packages: {e}")
        raise
    except Exception as e:
        log.error(f"Failed to create animation: {e}")
        raise


def toggle_mode_animation(
    analyzer,
    ax: Any,
    row_idx: int,
    col_idx: int,
    component: Union[str, int],
    z_layer: int,
) -> None:
    """
    Toggle animation for a mode cell (row_idx, col_idx).

    When ``analyzer.config.use_holography`` is *True* the whole column is
    animated via :func:`start_column_animation` (single ``FuncAnimation``,
    blit-optimised, GC-safe).  When holography is off, only the clicked row
    is animated as before.
    """
    if not ANIMATION_AVAILABLE:
        log.warning("Animation not available - matplotlib.animation required")
        return

    # Initialize animation tracking if needed
    if not hasattr(analyzer, "_mode_animations"):
        analyzer._mode_animations = {}
        analyzer._animated_axes = set()
    if not hasattr(analyzer, "_column_animations"):
        analyzer._column_animations = {}

    use_holo = getattr(analyzer.config, "use_holography", False)

    if use_holo:
        # ── Column-wide toggle ─────────────────────────────────────────────
        col_key = f"col_{col_idx}"
        col_running = col_key in analyzer._column_animations

        if col_running:
            stop_column_animation(analyzer, col_idx)
            # Redraw every row as static
            vis_types = []
            if analyzer.config.show_magnitude:
                vis_types.append("magnitude")
            if analyzer.config.show_phase:
                vis_types.append("phase")
            if analyzer.config.show_combined:
                vis_types.append("combined")
            for r, _ in enumerate(vis_types):
                if r < len(analyzer._mode_axes) and col_idx < len(analyzer._mode_axes[r]):
                    analyzer._update_single_mode_plot(
                        analyzer._mode_axes[r][col_idx], r, col_idx, component, z_layer
                    )
            analyzer._interactive_fig.canvas.draw()
            log.info(f"Stopped column animation for m_{component} (col {col_idx})")
        else:
            frames = analyzer.config.animation_time_steps
            start_column_animation(analyzer, col_idx, component, z_layer, frames=frames)
            analyzer._interactive_fig.canvas.draw_idle()
            log.info(f"Started column animation for m_{component} (col {col_idx})")

    else:
        # ── Legacy per-row toggle ──────────────────────────────────────────
        axis_key = (row_idx, col_idx)

        if axis_key in analyzer._animated_axes:
            stop_mode_animation(analyzer, axis_key)
            analyzer._update_single_mode_plot(ax, row_idx, col_idx, component, z_layer)
            analyzer._interactive_fig.canvas.draw()
            log.info(
                f"Stopped animation for m_{component} (row {row_idx}, col {col_idx})"
            )
        else:
            start_mode_animation(analyzer, ax, row_idx, col_idx, component, z_layer)
            log.info(
                f"Started animation for m_{component} (row {row_idx}, col {col_idx})"
            )



def stop_mode_animation(analyzer, axis_key: tuple[int, int]) -> None:
    """Stop animation for specific axis."""
    if axis_key in analyzer._mode_animations:
        anim = analyzer._mode_animations[axis_key]
        try:
            anim.event_source.stop()
            del analyzer._mode_animations[axis_key]
        except Exception as e:
            log.debug(f"Error stopping animation: {e}")

    analyzer._animated_axes.discard(axis_key)


def stop_column_animation(analyzer, col_idx: int) -> None:
    """Stop the column-wide animation for column *col_idx*."""
    col_key = f"col_{col_idx}"
    if not hasattr(analyzer, "_column_animations"):
        return
    if col_key in analyzer._column_animations:
        anim = analyzer._column_animations[col_key]
        try:
            anim.event_source.stop()
        except Exception as e:
            log.debug(f"Error stopping column animation: {e}")
        del analyzer._column_animations[col_key]

    # Remove axis keys for this column from the animated set
    to_remove = {k for k in analyzer._animated_axes if k[1] == col_idx}
    analyzer._animated_axes -= to_remove
    log.info(f"Stopped column animation for col {col_idx}")


def start_column_animation(
    analyzer,
    col_idx: int,
    component: Union[str, int],
    z_layer: int,
    frames: int = 60,
    interval_ms: int = 50,
) -> None:
    """
    Animate **all display rows** of one component column with a single
    ``FuncAnimation`` using Matplotlib blitting.

    This is the Micromag Holographic Engine design:

    * **Row 0 – Amplitude** is time-invariant and is rendered once as a frozen
      background.  The ``FuncAnimation`` touches it only to return it in the
      blit list (zero cost per frame).
    * **Row 1 – Holography / Phase** rotates the HSV hue by
      ``(H_base - t/(2π)) % 1.0`` – no trigonometry, just modular arithmetic.
    * **Row 2 – Real part** calls
      :py:meth:`~TopologicalAnimator.get_real_frame` – one complex multiply and
      ``np.real`` per frame with a fixed colour scale that never flickers.

    The returned ``FuncAnimation`` is stored on *analyzer* as
    ``analyzer._column_animations[f"col_{col_idx}"]``, preventing Python's
    garbage collector from destroying it.

    Parameters
    ----------
    analyzer :
        FMRModeAnalyzer instance (must have ``_mode_axes``, ``_interactive_fig``).
    col_idx : int
        Column index in the mode axes grid.
    component : str or int
        Magnetization component (``'x'``, ``'+'``, ``'rho'``, …).
    z_layer : int
        Simulation z-layer.
    frames : int
        Number of frames per full period (default 60).
    interval_ms : int
        Milliseconds between frames (default 50 → 20 fps).
    """
    if not ANIMATION_AVAILABLE:
        log.warning("Column animation not available – matplotlib.animation required")
        return

    try:
        from ..vortex_optics import TopologicalAnimator, VortexOptics

        # ── Guard: kill previous column animation if present ─────────────────
        if not hasattr(analyzer, "_column_animations"):
            analyzer._column_animations = {}
        stop_column_animation(analyzer, col_idx)

        # ── Fetch mode data ───────────────────────────────────────────────────
        mode_data = analyzer.get_mode(analyzer._current_frequency, z_layer)
        comp_data = mode_data.get_component(component)

        # ── Build TopologicalAnimator (one-time expensive math) ───────────────
        holo_gamma = getattr(analyzer.config, "holography_gamma", 0.5)
        holo_noise = getattr(analyzer.config, "holography_noise_threshold", 1e-4)
        use_holo = getattr(analyzer.config, "use_holography", False)

        topoanim = TopologicalAnimator(comp_data, gamma=holo_gamma, noise_threshold=holo_noise)

        # ── Determine active vis rows ─────────────────────────────────────────
        vis_types: list[str] = []
        if analyzer.config.show_magnitude:
            vis_types.append("magnitude")
        if analyzer.config.show_phase:
            vis_types.append("phase")
        if analyzer.config.show_combined:
            vis_types.append("combined")

        n_rows = len(vis_types)
        time_steps = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
        comp_label = VortexOptics.get_component_label(str(component), latex=True)

        # ── Draw initial frames and collect AxesImage handles ────────────────
        im_handles: dict[int, Any] = {}   # row_idx → AxesImage

        for row_idx, vis_type in enumerate(vis_types):
            if row_idx >= len(analyzer._mode_axes):
                break
            if col_idx >= len(analyzer._mode_axes[row_idx]):
                break

            ax = analyzer._mode_axes[row_idx][col_idx]
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

            common_kw = dict(
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                origin="lower",
            )

            if vis_type == "magnitude":
                # ── Row 0: Amplitude – render once, never update ──────────────
                im = ax.imshow(
                    np.abs(comp_data),
                    cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_magnitude),
                    **common_kw,
                )
                ax.set_title(f"|{comp_label}|", fontsize=8)

            elif vis_type == "phase":
                # ── Row 1: Holography or classical phase ──────────────────────
                if use_holo:
                    im = ax.imshow(
                        topoanim.get_hologram_frame(0.0),
                        **common_kw,
                    )
                    ax.set_title(f"Hologram {comp_label}", fontsize=8)
                else:
                    phase = np.angle(comp_data)
                    im = ax.imshow(
                        phase,
                        cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                        vmin=-np.pi,
                        vmax=np.pi,
                        **common_kw,
                    )
                    ax.set_title(f"arg({comp_label})", fontsize=8)

            elif vis_type == "combined":
                # ── Row 2: Real part with fixed colour scale ──────────────────
                im = ax.imshow(
                    topoanim.get_real_frame(0.0),
                    cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                    vmin=-topoanim.max_amp,
                    vmax=topoanim.max_amp,
                    **common_kw,
                )
                ax.set_title(f"Re[{comp_label}(t)]", fontsize=8)

            im_handles[row_idx] = im
            analyzer._animated_axes.add((row_idx, col_idx))

        if not im_handles:
            log.error(f"No axes found for col_idx={col_idx}")
            return

        # Pre-fetch static references so the closure is GC-safe
        _phase_arr = np.angle(comp_data) if use_holo is False else None

        # ── Main update function (blitting: only changed artists returned) ────
        def _update_column(frame_idx: int) -> list:
            t = time_steps[frame_idx]
            updated: list = []

            for row_idx, vis_type in enumerate(vis_types):
                if row_idx not in im_handles:
                    continue
                im = im_handles[row_idx]

                if vis_type == "magnitude":
                    # Amplitude is constant — return handle without touching data
                    # (blit still tracks it but redraws only on dirty)
                    pass

                elif vis_type == "phase":
                    if use_holo:
                        im.set_data(topoanim.get_hologram_frame(t))
                    else:
                        shifted = (_phase_arr + t) % (2.0 * np.pi)
                        shifted = np.where(shifted > np.pi, shifted - 2.0 * np.pi, shifted)
                        im.set_array(shifted)
                    updated.append(im)

                elif vis_type == "combined":
                    im.set_array(topoanim.get_real_frame(t))
                    updated.append(im)

            return updated

        # ── Launch animation ──────────────────────────────────────────────────
        anim = FuncAnimation(
            analyzer._interactive_fig,
            _update_column,
            frames=frames,
            interval=interval_ms,
            blit=True,
            repeat=True,
        )

        # ── Store to prevent garbage collection ───────────────────────────────
        col_key = f"col_{col_idx}"
        analyzer._column_animations[col_key] = anim
        log.info(
            f"Column animation started for col={col_idx} component={component} "
            f"({frames} frames, {interval_ms} ms interval, holography={use_holo})"
        )

    except Exception as e:
        log.error(f"Failed to start column animation for col={col_idx}: {e}")
        raise




def save_animated_view(analyzer, save_path: str, z_layer: int = 0) -> None:
    """Save current animated view to video file."""
    if not analyzer._mode_animations:
        raise ValueError("No active animations to save")

    # Import required modules
    try:
        from matplotlib.animation import PillowWriter
    except ImportError:
        raise ImportError("Animation saving requires matplotlib.animation")

    log.info(f"Creating animation with {len(analyzer._mode_animations)} animated modes")

    # Determine writer based on file extension
    file_ext = save_path.lower().split(".")[-1]

    if file_ext == "mp4":
        # Ensure FFmpeg is available
        ffmpeg_path = _ensure_ffmpeg_available()
        if ffmpeg_path:
            try:
                writer = _create_ffmpeg_writer(ffmpeg_path, fps=20, bitrate=1800)
                writer_name = "ffmpeg"
            except Exception as e:
                log.warning(
                    f"FFMpeg initialization failed: {e}, falling back to Pillow"
                )
                writer = PillowWriter(fps=10)
                writer_name = "pillow"
                save_path = save_path.replace(".mp4", ".gif")
        else:
            log.warning("FFMpeg not available, falling back to Pillow")
            writer = PillowWriter(fps=10)
            writer_name = "pillow"
            save_path = save_path.replace(".mp4", ".gif")

    elif file_ext == "gif":
        writer = PillowWriter(fps=10)
        writer_name = "pillow"
    elif file_ext == "avi":
        ffmpeg_path = _ensure_ffmpeg_available()
        if ffmpeg_path:
            try:
                writer = _create_ffmpeg_writer(ffmpeg_path, fps=20, bitrate=1800)
                writer_name = "ffmpeg"
            except Exception as e:
                log.warning(
                    f"FFMpeg initialization failed: {e}, falling back to GIF"
                )
                writer = PillowWriter(fps=10)
                writer_name = "pillow"
                save_path = save_path.replace(".avi", ".gif")
        else:
            log.warning("FFMpeg not available, falling back to GIF")
            writer = PillowWriter(fps=10)
            writer_name = "pillow"
            save_path = save_path.replace(".avi", ".gif")
    else:
        writer = PillowWriter(fps=10)
        writer_name = "pillow"
        save_path = save_path.replace(f".{file_ext}", ".gif")

    # Create master animation
    def animate_all_modes(frame):
        """Update all animated mode plots simultaneously"""
        try:
            time_step = (frame % 30) / 30.0 * 2 * np.pi

            for row_idx, col_idx in analyzer._animated_axes:
                try:
                    ax = analyzer._mode_axes[row_idx][col_idx]
                    mode_data = analyzer.get_mode(analyzer._current_frequency, z_layer)

                    if hasattr(analyzer, '_current_components') and col_idx < len(analyzer._current_components):
                        component = analyzer._current_components[col_idx]
                    else:
                        components_default = ["x", "y", "z"]
                        component = components_default[col_idx] if col_idx < len(components_default) else "z"
                    
                    comp_data = mode_data.get_component(component)

                    vis_types = []
                    if analyzer.config.show_magnitude:
                        vis_types.append("magnitude")
                    if analyzer.config.show_phase:
                        vis_types.append("phase")
                    if analyzer.config.show_combined:
                        vis_types.append("combined")

                    vis_type = vis_types[row_idx]

                    images = [
                        child
                        for child in ax.get_children()
                        if hasattr(child, "set_array")
                    ]
                    if images:
                        im = images[0]

                        if vis_type == "magnitude":
                            amplitude = np.abs(comp_data)
                            pulse = 0.8 + 0.2 * np.sin(time_step)
                            im.set_array(amplitude * pulse)
                        elif vis_type == "phase":
                            phase = np.angle(comp_data)
                            current_phase = (phase + time_step) % (2 * np.pi)
                            current_phase = np.where(
                                current_phase > np.pi,
                                current_phase - 2 * np.pi,
                                current_phase,
                            )
                            im.set_array(current_phase)
                        elif vis_type == "combined":
                            amplitude = np.abs(comp_data)
                            phase = np.angle(comp_data)
                            real_part = amplitude * np.cos(phase + time_step)
                            im.set_array(real_part)

                except Exception as e:
                    log.debug(f"Error updating animation for ({row_idx}, {col_idx}): {e}")

            return []

        except Exception as e:
            log.debug(f"Error in animate_all_modes: {e}")
            return []

    total_frames = 30

    anim = FuncAnimation(
        analyzer._interactive_fig,
        animate_all_modes,
        frames=total_frames,
        interval=50,
        blit=False,
        repeat=True,
    )

    log.info(f"Saving {total_frames} frames using {writer_name} writer...")

    try:
        anim.save(save_path, writer=writer, dpi=150)
        log.info("✅ Animation saved successfully!")
    except Exception as e:
        log.error(f"Failed to save animation: {e}")
        base_name = save_path.rsplit(".", 1)[0]
        for i in range(min(100, total_frames)):
            animate_all_modes(i)
            analyzer._interactive_fig.canvas.draw()
            static_path = f"{base_name}_frame_{i:03d}.png"
            analyzer._interactive_fig.savefig(static_path, dpi=150, bbox_inches="tight")

        log.info(f"Saved static frames to {base_name}_frame_*.png")
        raise RuntimeError(
            f"Could not save as {file_ext}, saved static frames instead"
        )


def start_mode_animation(
    analyzer,
    ax: Any,
    row_idx: int,
    col_idx: int,
    component: Union[str, int],
    z_layer: int,
) -> None:
    """
    Start in-place animation for a single mode axis cell (row, col).

    Uses :class:`~mmpp.fft.modes.vortex_optics.TopologicalAnimator` for
    holography and real-part rows: amplitude/phase maps are cached once at
    construction, so each frame costs only one modulo op (hue rotation) or
    one multiply (real projection) per pixel.

    Row semantics
    -------------
    * **magnitude** – static frame pulsed ±20 % (amplitude is time-invariant)
    * **phase**     – holographic HSV rotation when ``use_holography=True``,
                      otherwise classical phase-colormap shift
    * **combined**  – Re[m(r) · exp(-it)] with fixed vmin/vmax (no flicker)
    """
    try:
        from ..vortex_optics import TopologicalAnimator

        mode_data = analyzer.get_mode(analyzer._current_frequency, z_layer)
        comp_data = mode_data.get_component(component)

        vis_types = []
        if analyzer.config.show_magnitude:
            vis_types.append("magnitude")
        if analyzer.config.show_phase:
            vis_types.append("phase")
        if analyzer.config.show_combined:
            vis_types.append("combined")

        if row_idx >= len(vis_types):
            log.error(f"Invalid row index {row_idx} for visualization types")
            return

        vis_type = vis_types[row_idx]

        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

        frames = analyzer.config.animation_time_steps
        interval_ms = 50  # 50 ms → 20 fps target

        # ── Magnitude row ────────────────────────────────────────────────────
        if vis_type == "magnitude":
            amplitude = np.abs(comp_data)
            time_steps = np.linspace(0, 2 * np.pi, frames)

            im = ax.imshow(
                amplitude,
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_magnitude),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                origin="lower",
            )
            ax.set_title(f"|m_{component}|")

            # Amplitude is time-invariant; gentle pulse only for visual feedback
            def animate_magnitude(frame, _amp=amplitude, _ts=time_steps, _im=im):
                pulse = 0.85 + 0.15 * np.sin(_ts[frame])
                _im.set_array(_amp * pulse)
                return [_im]

            anim = FuncAnimation(
                analyzer._interactive_fig,
                animate_magnitude,
                frames=frames,
                interval=interval_ms,
                blit=True,
                repeat=True,
            )

        # ── Phase / holography row ────────────────────────────────────────────
        elif vis_type == "phase":
            use_holo = getattr(analyzer.config, "use_holography", False)
            holo_gamma = getattr(analyzer.config, "holography_gamma", 0.5)
            holo_noise = getattr(analyzer.config, "holography_noise_threshold", 1e-4)

            if use_holo:
                # Build animator once (caches amp + phase; per-frame = modulo)
                topoanim = TopologicalAnimator(
                    comp_data, gamma=holo_gamma, noise_threshold=holo_noise
                )
                time_steps = np.linspace(0, 2 * np.pi, frames)

                im = ax.imshow(
                    topoanim.get_hologram_frame(0.0),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=analyzer.config.interpolation,
                    origin="lower",
                )
                ax.set_title(f"Hologram(m_{component})")

                def animate_holography(frame, _ta=topoanim, _ts=time_steps, _im=im):
                    _im.set_data(_ta.get_hologram_frame(_ts[frame]))
                    return [_im]

                anim = FuncAnimation(
                    analyzer._interactive_fig,
                    animate_holography,
                    frames=frames,
                    interval=interval_ms,
                    blit=True,
                    repeat=True,
                )

            else:
                # Standard cyclic phase-colormap animation
                phase = np.angle(comp_data)
                time_steps = np.linspace(0, 2 * np.pi, frames)

                im = ax.imshow(
                    phase,
                    cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=analyzer.config.interpolation,
                    vmin=-np.pi,
                    vmax=np.pi,
                    origin="lower",
                )
                ax.set_title(f"arg(m_{component})")

                def animate_phase(frame, _ph=phase, _ts=time_steps, _im=im):
                    shifted = (_ph + _ts[frame]) % (2 * np.pi)
                    shifted = np.where(shifted > np.pi, shifted - 2 * np.pi, shifted)
                    _im.set_array(shifted)
                    return [_im]

                anim = FuncAnimation(
                    analyzer._interactive_fig,
                    animate_phase,
                    frames=frames,
                    interval=interval_ms,
                    blit=True,
                    repeat=True,
                )

        # ── Real-part / combined row ──────────────────────────────────────────
        elif vis_type == "combined":
            holo_gamma = getattr(analyzer.config, "holography_gamma", 0.5)
            holo_noise = getattr(analyzer.config, "holography_noise_threshold", 1e-4)
            topoanim = TopologicalAnimator(
                comp_data, gamma=holo_gamma, noise_threshold=holo_noise
            )
            time_steps = np.linspace(0, 2 * np.pi, frames)

            # Fixed colour scale derived from max amplitude → no flicker!
            vmax = topoanim.max_amp

            im = ax.imshow(
                topoanim.get_real_frame(0.0),
                cmap=analyzer.config._resolve_colormap(analyzer.config.colormap_phase),
                extent=mode_data.extent,
                aspect="equal",
                interpolation=analyzer.config.interpolation,
                vmin=-vmax,
                vmax=vmax,
                origin="lower",
            )
            ax.set_title(f"Re[m_{component}(t)]")

            def animate_combined(frame, _ta=topoanim, _ts=time_steps, _im=im):
                _im.set_array(_ta.get_real_frame(_ts[frame]))
                return [_im]

            anim = FuncAnimation(
                analyzer._interactive_fig,
                animate_combined,
                frames=frames,
                interval=interval_ms,
                blit=True,
                repeat=True,
            )

        axis_key = (row_idx, col_idx)
        analyzer._mode_animations[axis_key] = anim
        analyzer._animated_axes.add(axis_key)

    except Exception as e:
        log.error(f"Failed to start mode animation: {e}")
