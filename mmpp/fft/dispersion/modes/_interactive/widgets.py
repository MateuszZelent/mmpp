"""
Widget creation and layout management for InteractiveDispersionModes.

Provides builders for ipywidgets-based UI components.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

try:
    import ipywidgets as widgets

    _HAS_WIDGETS = True
except ImportError:
    _HAS_WIDGETS = False
    widgets = None  # type: ignore

if TYPE_CHECKING:
    pass


class WidgetBuilder:
    """Handles creation of ipywidgets for interactive dispersion analysis."""

    def __init__(self, default_params: dict[str, Any]):
        """
        Initialize widget builder.

        Parameters
        ----------
        default_params : dict
            Default parameter values for widgets.
        """
        if not _HAS_WIDGETS:
            raise ImportError("ipywidgets required for interactive mode")

        self.params = default_params
        self.widgets: dict[str, Any] = {}

    def create_all_widgets(self) -> dict[str, Any]:
        """
        Create all widgets for the interactive interface.

        Returns
        -------
        dict
            Dictionary of created widgets by name.
        """
        params = self.params

        # Lattice parameters
        self.widgets["lattice"] = widgets.FloatSlider(
            value=params["lattice_nm"],
            min=50,
            max=2000,
            step=5,
            description="a [nm]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.widgets["n_bz_mask"] = widgets.IntSlider(
            value=params["n_bz_mask"],
            min=1,
            max=10,
            step=1,
            description="N_BZ mask:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        # Frequency range
        self.widgets["fmin"] = widgets.FloatSlider(
            value=params["f_min_ghz"],
            min=0,
            max=params["f_max_ghz"],
            step=0.1,
            description="f min [GHz]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.widgets["fmax"] = widgets.FloatSlider(
            value=params["f_max_ghz"],
            min=0.1,
            max=params["f_max_ghz"] * 1.5,
            step=0.1,
            description="f max [GHz]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        # k-direction selection
        self.widgets["k_direction"] = widgets.Dropdown(
            options=[
                ("Both ±k", "both"),
                ("Only +k", "positive"),
                ("Only -k", "negative"),
            ],
            value=params["k_direction"],
            description="k-dirs:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        # Colormaps
        self.widgets["cmap_disp"] = widgets.Dropdown(
            options=["viridis", "plasma", "cividis", "turbo", "inferno"],
            value=params["cmap_disp"],
            description="Cmap disp:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        self.widgets["cmap_mode"] = widgets.Dropdown(
            options=["RdBu_r", "seismic", "coolwarm", "bwr", "PiYG"],
            value=params.get("cmap_mode", "RdBu_r"),
            description="Cmap mode:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        # Buttons
        self.widgets["update"] = widgets.Button(
            description="🔄 Update",
            button_style="success",
            layout=widgets.Layout(width="95%"),
        )

        self.widgets["auto_detect"] = widgets.Button(
            description="🔍 Auto-detect a",
            button_style="info",
            layout=widgets.Layout(width="95%"),
        )

        # Info displays
        self.widgets["info"] = widgets.HTML(
            value="<small>Click on dispersion to select mode (k, f)</small>",
        )

        self.widgets["mode_info"] = widgets.HTML(
            value="<small>No mode selected</small>",
        )

        # Output area for plots
        self.widgets["output"] = widgets.Output(
            layout=widgets.Layout(width="100%", height="auto")
        )

        return self.widgets

    def create_layout(self, widgets_dict: dict[str, Any]) -> widgets.Widget:
        """
        Create the main layout with controls and plot area.

        Parameters
        ----------
        widgets_dict : dict
            Dictionary of widgets created by create_all_widgets().

        Returns
        -------
        widgets.Widget
            Complete layout widget ready for display.
        """
        # Left panel: controls
        left_panel = widgets.VBox(
            [
                widgets.HTML("<b>🌊 BZ Mode Analysis</b>"),
                widgets.HTML("<hr style='margin:2px'>"),
                widgets.HTML("<small><b>Lattice</b></small>"),
                widgets_dict["lattice"],
                widgets_dict["auto_detect"],
                widgets.HTML("<small><b>Mask Settings</b></small>"),
                widgets_dict["n_bz_mask"],
                widgets_dict["k_direction"],
                widgets.HTML("<small><b>Frequency Range</b></small>"),
                widgets_dict["fmin"],
                widgets_dict["fmax"],
                widgets.HTML("<small><b>Display</b></small>"),
                widgets_dict["cmap_disp"],
                widgets_dict["cmap_mode"],
                widgets_dict["update"],
                widgets.HTML("<hr style='margin:5px'>"),
                widgets_dict["info"],
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Selected Mode</b></small>"),
                widgets_dict["mode_info"],
            ],
            layout=widgets.Layout(
                width="200px",
                padding="5px",
                border="1px solid #ddd",
            ),
        )

        # Right panel: stacked plots
        right_panel = widgets.VBox(
            [
                widgets_dict["output"],
            ],
            layout=widgets.Layout(
                width="calc(100% - 220px)",
                min_width="700px",
            ),
        )

        # Main layout
        main = widgets.HBox(
            [
                left_panel,
                right_panel,
            ],
            layout=widgets.Layout(
                width="100%",
            ),
        )

        return main

    def connect_callbacks(
        self,
        widgets_dict: dict[str, Any],
        on_update: Callable[..., Any],
        on_auto_detect: Callable[..., Any],
        on_display_change: Callable[..., Any],
        on_mode_change: Callable[..., Any],
    ) -> None:
        """
        Connect widget callbacks to handler functions.

        Parameters
        ----------
        widgets_dict : dict
            Dictionary of widgets
        on_update : callable
            Callback for update button
        on_auto_detect : callable
            Callback for auto-detect button
        on_display_change : callable
            Callback for display parameter changes
        on_mode_change : callable
            Callback for mode parameter changes
        """
        # Button callbacks
        widgets_dict["update"].on_click(on_update)
        widgets_dict["auto_detect"].on_click(on_auto_detect)

        # Display parameter changes (immediate update)
        for key in ["cmap_disp", "fmin", "fmax"]:
            widgets_dict[key].observe(on_display_change, names="value")

        # Mode visualization parameter changes
        for key in ["n_bz_mask", "k_direction", "cmap_mode"]:
            widgets_dict[key].observe(on_mode_change, names="value")
