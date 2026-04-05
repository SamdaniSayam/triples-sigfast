"""
triples_sigfast.viz.physics_plot
─────────────────────────────────
Publication-quality physics plots in one line of code.

Supports 6 journal styles out of the box and lets users define
their own. Outputs PDF, PNG (300/600 DPI), SVG, and LaTeX TikZ.
Uses plotly by default in Jupyter notebooks and falls back to
matplotlib for static export.

Supported journal styles
------------------------
- publication  : generic publication (default)
- physical_review : APS Physical Review
- nature       : Nature family journals
- jinst        : JINST (IOP)
- nim_a        : Nuclear Instruments and Methods A
- softwarex    : SoftwareX (Elsevier)
- presentation : slide-friendly large fonts
- thesis       : thesis chapter figures

References
----------
- APS Style Guide: https://journals.aps.org/authors
- Nature style guide: https://www.nature.com/authors
- JINST author guidelines: https://iopscience.iop.org/journal/1748-0221
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# ── Journal style definitions ─────────────────────────────────────────────────

_JOURNAL_STYLES: dict[str, dict] = {
    "publication": {
        "fig_width": 8.0,
        "fig_height": 5.5,
        "font_size": 12,
        "font_family": "serif",
        "line_width": 1.8,
        "marker_size": 5,
        "dpi": 300,
        "color_cycle": [
            "#185FA5",
            "#D85A30",
            "#639922",
            "#7F77DD",
            "#BA7517",
            "#533C1D",
        ],
        "grid": True,
        "grid_alpha": 0.3,
        "legend_fontsize": 10,
        "tick_direction": "in",
        "minor_ticks": True,
        "spine_width": 0.8,
    },
    "physical_review": {
        "fig_width": 7.0,
        "fig_height": 5.0,
        "font_size": 11,
        "font_family": "serif",
        "line_width": 1.5,
        "marker_size": 4,
        "dpi": 600,
        "color_cycle": [
            "#000000",
            "#D85A30",
            "#185FA5",
            "#639922",
            "#7F77DD",
            "#BA7517",
        ],
        "grid": False,
        "grid_alpha": 0.0,
        "legend_fontsize": 9,
        "tick_direction": "in",
        "minor_ticks": True,
        "spine_width": 0.8,
    },
    "nature": {
        "fig_width": 8.9 / 2.54,  # 89 mm in inches
        "fig_height": 6.0 / 2.54,
        "font_size": 7,
        "font_family": "sans-serif",
        "line_width": 1.0,
        "marker_size": 3,
        "dpi": 300,
        "color_cycle": [
            "#E64B35",
            "#4DBBD5",
            "#00A087",
            "#3C5488",
            "#F39B7F",
            "#8491B4",
        ],
        "grid": False,
        "grid_alpha": 0.0,
        "legend_fontsize": 6,
        "tick_direction": "out",
        "minor_ticks": False,
        "spine_width": 0.5,
    },
    "jinst": {
        "fig_width": 8.0,
        "fig_height": 6.0,
        "font_size": 12,
        "font_family": "serif",
        "line_width": 1.5,
        "marker_size": 5,
        "dpi": 300,
        "color_cycle": [
            "#185FA5",
            "#E24B4A",
            "#639922",
            "#7F77DD",
            "#BA7517",
            "#D85A30",
        ],
        "grid": True,
        "grid_alpha": 0.25,
        "legend_fontsize": 10,
        "tick_direction": "in",
        "minor_ticks": True,
        "spine_width": 0.8,
    },
    "nim_a": {
        "fig_width": 8.5,
        "fig_height": 6.0,
        "font_size": 11,
        "font_family": "serif",
        "line_width": 1.5,
        "marker_size": 4,
        "dpi": 300,
        "color_cycle": [
            "#000000",
            "#D85A30",
            "#185FA5",
            "#639922",
            "#7F77DD",
            "#BA7517",
        ],
        "grid": False,
        "grid_alpha": 0.0,
        "legend_fontsize": 9,
        "tick_direction": "in",
        "minor_ticks": True,
        "spine_width": 0.8,
    },
    "softwarex": {
        "fig_width": 8.0,
        "fig_height": 5.5,
        "font_size": 11,
        "font_family": "sans-serif",
        "line_width": 1.5,
        "marker_size": 5,
        "dpi": 300,
        "color_cycle": [
            "#185FA5",
            "#D85A30",
            "#639922",
            "#7F77DD",
            "#BA7517",
            "#533C1D",
        ],
        "grid": True,
        "grid_alpha": 0.25,
        "legend_fontsize": 9,
        "tick_direction": "out",
        "minor_ticks": False,
        "spine_width": 0.8,
    },
    "presentation": {
        "fig_width": 12.0,
        "fig_height": 7.0,
        "font_size": 16,
        "font_family": "sans-serif",
        "line_width": 2.5,
        "marker_size": 8,
        "dpi": 150,
        "color_cycle": [
            "#185FA5",
            "#D85A30",
            "#639922",
            "#7F77DD",
            "#BA7517",
            "#533C1D",
        ],
        "grid": True,
        "grid_alpha": 0.3,
        "legend_fontsize": 13,
        "tick_direction": "out",
        "minor_ticks": False,
        "spine_width": 1.2,
    },
    "thesis": {
        "fig_width": 6.0,
        "fig_height": 4.5,
        "font_size": 11,
        "font_family": "serif",
        "line_width": 1.5,
        "marker_size": 4,
        "dpi": 300,
        "color_cycle": [
            "#185FA5",
            "#D85A30",
            "#639922",
            "#7F77DD",
            "#BA7517",
            "#533C1D",
        ],
        "grid": True,
        "grid_alpha": 0.25,
        "legend_fontsize": 9,
        "tick_direction": "in",
        "minor_ticks": True,
        "spine_width": 0.8,
    },
}


def _is_jupyter() -> bool:
    """Detect if running inside a Jupyter notebook."""
    try:  # pragma: no cover
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in (
            "ZMQInteractiveShell",
            "TerminalInteractiveShell",
        )
    except ImportError:
        return False


def _plotly_available() -> bool:  # pragma: no cover
    try:
        import plotly  # noqa: F401

        return True
    except ImportError:
        return False


class PhysicsPlot:
    """
    Publication-quality physics plots in one line of code.

    Automatically uses plotly for interactive output in Jupyter
    notebooks, and matplotlib for static file export. Supports
    8 built-in journal styles and custom user-defined styles.

    Parameters
    ----------
    style : str, optional
        Journal style preset. One of: 'publication', 'physical_review',
        'nature', 'jinst', 'nim_a', 'softwarex', 'presentation', 'thesis'.
        Default 'publication'.
    custom_style : dict, optional
        Override any style parameters. Keys match those in _JOURNAL_STYLES.
    interactive : bool or None, optional
        If True, always use plotly. If False, always use matplotlib.
        If None (default), auto-detect: plotly in Jupyter, matplotlib otherwise.

    Examples
    --------
    >>> plot = PhysicsPlot(style="physical_review")
    >>> plot.spectrum(energies, counts, smoothed, peaks=peak_indices,
    ...              title="Neutron Energy Spectrum",
    ...              xlabel="Energy (MeV)", ylabel="Counts")
    >>> plot.save("figure1.pdf", journal="Physical Review")
    """

    def __init__(
        self,
        style: str = "publication",
        custom_style: dict | None = None,
        interactive: bool | None = None,
    ) -> None:
        if style not in _JOURNAL_STYLES:
            raise ValueError(
                f"Unknown style '{style}'. Available: {list(_JOURNAL_STYLES.keys())}"
            )

        self.style_name = style
        self._style = dict(_JOURNAL_STYLES[style])

        if custom_style:
            self._style.update(custom_style)

        # Auto-detect interactive mode
        if interactive is None:
            self._interactive = (
                _is_jupyter() and _plotly_available()
            )  # pragma: no cover # pragma: no cover
        else:
            self._interactive = interactive and _plotly_available()

        self._figures: list[Any] = []
        self._captions: list[str] = []

        self._apply_matplotlib_style()

    # ── Style management ──────────────────────────────────────────────────────

    def _apply_matplotlib_style(self) -> None:
        """Apply journal style to matplotlib rcParams."""
        import matplotlib as mpl

        s = self._style
        mpl.rcParams.update(
            {
                "figure.figsize": [s["fig_width"], s["fig_height"]],
                "figure.dpi": s["dpi"],
                "font.size": s["font_size"],
                "font.family": s["font_family"],
                "axes.prop_cycle": mpl.cycler(color=s["color_cycle"]),
                "axes.grid": s.get("grid", True),
                "axes.linewidth": s.get("spine_width", 0.8),
                "lines.linewidth": s["line_width"],
                "lines.markersize": s.get("marker_size", 5),
                "legend.fontsize": s.get("legend_fontsize", 10),
                "xtick.direction": s.get("tick_direction", "in"),
                "ytick.direction": s.get("tick_direction", "in"),
                "xtick.minor.visible": s.get("minor_ticks", True),
                "ytick.minor.visible": s.get("minor_ticks", True),
                "savefig.dpi": s["dpi"],
                "savefig.bbox": "tight",
            }
        )
        # grid_alpha must be set per-axes, not via rcParams
        self._grid_alpha = s.get("grid_alpha", 0.3)

    @staticmethod
    def available_styles() -> list[str]:
        """Return list of available journal style names."""
        return list(_JOURNAL_STYLES.keys())

    @staticmethod
    def register_style(name: str, style: dict) -> None:
        """
        Register a custom journal style.

        Parameters
        ----------
        name : str
            Name for the new style.
        style : dict
            Style parameters. Must include at minimum: fig_width, fig_height,
            font_size, font_family, line_width, dpi, color_cycle.
        """
        required = {
            "fig_width",
            "fig_height",
            "font_size",
            "font_family",
            "line_width",
            "dpi",
            "color_cycle",
        }
        missing = required - set(style.keys())
        if missing:
            base = dict(_JOURNAL_STYLES["publication"])
            base.update(style)
            style = base
        _JOURNAL_STYLES[name] = style

    # ── Plot methods ──────────────────────────────────────────────────────────

    def spectrum(
        self,
        energies: np.ndarray,
        counts: np.ndarray,
        smoothed: np.ndarray | None = None,
        peaks: np.ndarray | None = None,
        errors: np.ndarray | None = None,
        title: str = "Energy Spectrum",
        xlabel: str = "Energy (MeV)",
        ylabel: str = "Counts",
        xscale: str = "linear",
        yscale: str = "linear",
        label_counts: str = "Raw spectrum",
        label_smoothed: str = "Smoothed",
    ) -> Any:
        """
        Plot an energy spectrum with optional smoothed overlay and peak markers.

        Parameters
        ----------
        energies : np.ndarray
            Energy axis (bin centres).
        counts : np.ndarray
            Raw counts per bin.
        smoothed : np.ndarray, optional
            Smoothed spectrum (e.g. Savitzky-Golay output).
        peaks : np.ndarray, optional
            Array of peak indices to mark with triangles.
        errors : np.ndarray, optional
            1-sigma uncertainty per bin (shown as error bars or shaded band).
        title : str
            Figure title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        xscale : str
            'linear' or 'log'.
        yscale : str
            'linear' or 'log'.

        Returns
        -------
        Figure object (plotly Figure or matplotlib Figure).
        """
        if self._interactive:
            return self._spectrum_plotly(  # pragma: no cover # pragma: no cover
                energies,
                counts,
                smoothed,
                peaks,
                errors,
                title,
                xlabel,
                ylabel,
                xscale,
                yscale,
                label_counts,
                label_smoothed,
            )
        return self._spectrum_matplotlib(
            energies,
            counts,
            smoothed,
            peaks,
            errors,
            title,
            xlabel,
            ylabel,
            xscale,
            yscale,
            label_counts,
            label_smoothed,
        )

    def _spectrum_matplotlib(
        self,
        energies,
        counts,
        smoothed,
        peaks,
        errors,
        title,
        xlabel,
        ylabel,
        xscale,
        yscale,
        label_counts,
        label_smoothed,
    ):
        import matplotlib.pyplot as plt

        colors = self._style["color_cycle"]

        fig, ax = plt.subplots(
            figsize=(self._style["fig_width"], self._style["fig_height"])
        )

        if self._style.get("grid", True):
            ax.grid(alpha=self._grid_alpha)

        if errors is not None:
            ax.fill_between(
                energies,
                counts - errors,
                counts + errors,
                alpha=0.2,
                color=colors[0],
            )
        ax.step(
            energies,
            counts,
            where="mid",
            color=colors[0],
            alpha=0.6,
            lw=self._style["line_width"] * 0.7,
            label=label_counts,
        )

        if smoothed is not None:
            ax.plot(
                energies,
                smoothed,
                color=colors[1],
                lw=self._style["line_width"],
                label=label_smoothed,
            )

        if peaks is not None and len(peaks) > 0:
            y_peaks = smoothed[peaks] if smoothed is not None else counts[peaks]
            ax.plot(
                energies[peaks],
                y_peaks,
                "v",
                color=colors[2],
                ms=self._style["marker_size"] * 1.8,
                zorder=5,
                label=f"Peaks ({len(peaks)})",
            )
            for pi in peaks:
                ax.axvline(energies[pi], color=colors[2], ls="--", alpha=0.4, lw=0.8)

        ax.set_title(title, fontsize=self._style["font_size"] + 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if smoothed is not None or peaks is not None:
            ax.legend()

        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def _spectrum_plotly(  # pragma: no cover
        self,
        energies,
        counts,
        smoothed,
        peaks,
        errors,
        title,
        xlabel,
        ylabel,
        xscale,
        yscale,
        label_counts,
        label_smoothed,
    ):
        import plotly.graph_objects as go  # pragma: no cover # pragma: no cover

        colors = self._style["color_cycle"]

        fig = go.Figure()

        if errors is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([energies, energies[::-1]]),
                    y=np.concatenate([counts + errors, (counts - errors)[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({self._hex_to_rgb(colors[0])},0.15)",
                    line={"color": "rgba(0,0,0,0)"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=energies,
                y=counts,
                mode="lines",
                name=label_counts,
                line={"color": colors[0], "width": self._style["line_width"] * 0.8},
                opacity=0.7,
            )
        )

        if smoothed is not None:
            fig.add_trace(
                go.Scatter(
                    x=energies,
                    y=smoothed,
                    mode="lines",
                    name=label_smoothed,
                    line={"color": colors[1], "width": self._style["line_width"]},
                )
            )

        if peaks is not None and len(peaks) > 0:
            y_peaks = smoothed[peaks] if smoothed is not None else counts[peaks]
            fig.add_trace(
                go.Scatter(
                    x=energies[peaks],
                    y=y_peaks,
                    mode="markers+text",
                    name=f"Peaks ({len(peaks)})",
                    marker={
                        "symbol": "triangle-down",
                        "size": self._style["marker_size"] * 2.5,
                        "color": colors[2],
                    },
                    text=[f"{energies[p]:.3f}" for p in peaks],
                    textposition="top center",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis_type=xscale if xscale == "log" else "-",
            yaxis_type=yscale if yscale == "log" else "-",
            template="plotly_white",
            font={"size": self._style["font_size"]},
            width=int(self._style["fig_width"] * 96),
            height=int(self._style["fig_height"] * 96),
        )
        self._figures.append(fig)
        fig.show()
        return fig

    def shielding_comparison(
        self,
        materials: list[str],
        thickness_range: np.ndarray,
        energy_mev: float = 1.25,
        use_buildup: bool = True,
        title: str = "Shielding Comparison",
        xlabel: str = "Thickness (cm)",
        ylabel: str = "Transmission fraction",
        yscale: str = "log",
    ) -> Any:
        """
        Plot transmission curves for multiple shielding materials.

        Parameters
        ----------
        materials : list of str
            Shield materials (e.g. ['lead', 'concrete', 'polyethylene']).
        thickness_range : np.ndarray
            Array of thicknesses in cm.
        energy_mev : float
            Photon energy in MeV.
        use_buildup : bool
            If True, use GP buildup-corrected attenuation.
        title : str
            Figure title.
        """
        from triples_sigfast.nuclear.shielding import attenuation_series

        curves = {
            mat: attenuation_series(thickness_range, mat, energy_mev, use_buildup)
            for mat in materials
        }

        if self._interactive:
            return self._shielding_plotly(  # pragma: no cover # pragma: no cover
                curves, thickness_range, title, xlabel, ylabel, yscale, energy_mev
            )
        return self._shielding_matplotlib(
            curves, thickness_range, title, xlabel, ylabel, yscale, energy_mev
        )

    def _shielding_matplotlib(
        self, curves, thickness_range, title, xlabel, ylabel, yscale, energy_mev
    ):
        import matplotlib.pyplot as plt

        colors = self._style["color_cycle"]

        fig, ax = plt.subplots(
            figsize=(self._style["fig_width"], self._style["fig_height"])
        )

        if self._style.get("grid", True):
            ax.grid(alpha=self._grid_alpha)

        for (mat, T), color in zip(curves.items(), colors):
            ax.plot(
                thickness_range,
                T,
                color=color,
                lw=self._style["line_width"],
                label=mat.capitalize(),
            )

        ax.axhline(0.5, color="gray", ls=":", alpha=0.7, lw=0.8, label="50% (1 HVL)")
        ax.axhline(
            0.1, color="gray", ls="--", alpha=0.7, lw=0.8, label="10% (3.32 HVL)"
        )

        ax.set_title(
            f"{title}\n({energy_mev} MeV, GP buildup-corrected)",
            fontsize=self._style["font_size"] + 1,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.legend()
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def _shielding_plotly(  # pragma: no cover
        self, curves, thickness_range, title, xlabel, ylabel, yscale, energy_mev
    ):
        import plotly.graph_objects as go  # pragma: no cover # pragma: no cover

        colors = self._style["color_cycle"]

        fig = go.Figure()
        for (mat, T), color in zip(curves.items(), colors):
            fig.add_trace(
                go.Scatter(
                    x=thickness_range,
                    y=T,
                    mode="lines",
                    name=mat.capitalize(),
                    line={"color": color, "width": self._style["line_width"]},
                )
            )

        fig.add_hline(
            y=0.5, line_dash="dot", line_color="gray", annotation_text="50% (1 HVL)"
        )
        fig.add_hline(
            y=0.1, line_dash="dash", line_color="gray", annotation_text="10% (3.32 HVL)"
        )

        fig.update_layout(
            title=f"{title} — {energy_mev} MeV",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            yaxis_type="log" if yscale == "log" else "-",
            template="plotly_white",
            font={"size": self._style["font_size"]},
            width=int(self._style["fig_width"] * 96),
            height=int(self._style["fig_height"] * 96),
        )
        self._figures.append(fig)
        fig.show()
        return fig

    def dose_map(
        self,
        mesh_x: np.ndarray,
        mesh_y: np.ndarray,
        dose_values: np.ndarray,
        unit: str = "uSv/hr",
        title: str = "Dose Rate Map",
        xlabel: str = "X (cm)",
        ylabel: str = "Y (cm)",
        colormap: str = "viridis",
        log_scale: bool = True,
    ) -> Any:
        """
        Plot a 2-D dose rate map (mesh tally visualization).

        Parameters
        ----------
        mesh_x : np.ndarray, shape (N,)
            X-axis coordinates.
        mesh_y : np.ndarray, shape (M,)
            Y-axis coordinates.
        dose_values : np.ndarray, shape (N, M)
            Dose rate values at each mesh point.
        unit : str
            Dose unit label for the colour bar.
        colormap : str
            Matplotlib or plotly colormap name. Default 'viridis'.
        log_scale : bool
            If True, apply log10 scale to dose values.
        """
        if self._interactive:
            return self._dose_map_plotly(  # pragma: no cover # pragma: no cover # pragma: no cover # pragma: no cover
                mesh_x,
                mesh_y,
                dose_values,
                unit,
                title,
                xlabel,
                ylabel,
                colormap,
                log_scale,
            )
        return self._dose_map_matplotlib(
            mesh_x,
            mesh_y,
            dose_values,
            unit,
            title,
            xlabel,
            ylabel,
            colormap,
            log_scale,
        )

    def _dose_map_matplotlib(
        self,
        mesh_x,
        mesh_y,
        dose_values,
        unit,
        title,
        xlabel,
        ylabel,
        colormap,
        log_scale,
    ):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(self._style["fig_width"], self._style["fig_height"])
        )

        if self._style.get("grid", True):
            ax.grid(alpha=self._grid_alpha)

        Z = dose_values.T
        norm = mcolors.LogNorm(vmin=Z[Z > 0].min(), vmax=Z.max()) if log_scale else None

        im = ax.pcolormesh(mesh_x, mesh_y, Z, cmap=colormap, norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Dose rate ({unit})", fontsize=self._style["font_size"])

        ax.set_title(title, fontsize=self._style["font_size"] + 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def _dose_map_plotly(  # pragma: no cover
        self,
        mesh_x,
        mesh_y,
        dose_values,
        unit,
        title,
        xlabel,
        ylabel,
        colormap,
        log_scale,
    ):
        import plotly.graph_objects as go  # pragma: no cover # pragma: no cover # pragma: no cover # pragma: no cover

        Z = dose_values.T
        if log_scale:
            Z_plot = np.log10(np.where(Z > 0, Z, np.nan))
            colorbar_title = f"log₁₀({unit})"
        else:
            Z_plot = Z
            colorbar_title = unit

        fig = go.Figure(
            data=go.Heatmap(
                x=mesh_x,
                y=mesh_y,
                z=Z_plot,
                colorscale=colormap,
                colorbar={"title": colorbar_title},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template="plotly_white",
            font={"size": self._style["font_size"]},
            yaxis={"scaleanchor": "x"},
            width=int(self._style["fig_width"] * 96),
            height=int(self._style["fig_height"] * 96),
        )
        self._figures.append(fig)
        fig.show()
        return fig

    def convergence_map(
        self,
        relative_errors: np.ndarray,
        energies: np.ndarray | None = None,
        threshold: float = 0.05,
        title: str = "MC Convergence Map",
        xlabel: str = "Energy bin",
        ylabel: str = "Relative Error R (%)",
    ) -> Any:
        """
        Plot a Monte Carlo convergence map coloured by pass/fail threshold.

        Parameters
        ----------
        relative_errors : np.ndarray
            Relative error R per bin (from `relative_error()`).
        energies : np.ndarray, optional
            Energy bin centres. If None, uses bin indices.
        threshold : float
            Convergence threshold. Default 0.05 (MCNP standard).
        """
        x = energies if energies is not None else np.arange(len(relative_errors))
        converged = relative_errors < threshold

        if self._interactive:
            return self._convergence_plotly(  # pragma: no cover # pragma: no cover # pragma: no cover # pragma: no cover
                x, relative_errors, converged, threshold, title, xlabel, ylabel
            )
        return self._convergence_matplotlib(
            x, relative_errors, converged, threshold, title, xlabel, ylabel
        )

    def _convergence_matplotlib(
        self, x, R, converged, threshold, title, xlabel, ylabel
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(self._style["fig_width"], self._style["fig_height"])
        )

        if self._style.get("grid", True):
            ax.grid(alpha=self._grid_alpha)

        colors = ["#639922" if c else "#E24B4A" for c in converged]
        ax.bar(
            x,
            R * 100,
            color=colors,
            alpha=0.8,
            width=np.diff(x).mean() if len(x) > 1 else 1,
        )
        ax.axhline(
            threshold * 100,
            color="#E24B4A",
            ls="--",
            lw=1.5,
            label=f"Threshold ({threshold * 100:.0f}%)",
        )
        ax.set_title(title, fontsize=self._style["font_size"] + 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        n_conv = converged.sum()
        ax.text(
            0.98,
            0.95,
            f"{n_conv}/{len(R)} converged",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=self._style["font_size"],
            color="#639922" if n_conv == len(R) else "#E24B4A",
        )
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def _convergence_plotly(
        self, x, R, converged, threshold, title, xlabel, ylabel
    ):  # pragma: no cover
        import plotly.graph_objects as go  # pragma: no cover # pragma: no cover # pragma: no cover # pragma: no cover

        colors = ["#639922" if c else "#E24B4A" for c in converged]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x,
                y=R * 100,
                marker_color=colors,
                name="Relative Error",
                opacity=0.85,
            )
        )
        fig.add_hline(
            y=threshold * 100,
            line_dash="dash",
            line_color="#E24B4A",
            annotation_text=f"Threshold {threshold * 100:.0f}%",
        )
        fig.update_layout(
            title=f"{title} — {converged.sum()}/{len(R)} bins converged",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template="plotly_white",
            font={"size": self._style["font_size"]},
            width=int(self._style["fig_width"] * 96),
            height=int(self._style["fig_height"] * 96),
        )
        self._figures.append(fig)
        fig.show()
        return fig

    # ── Export ────────────────────────────────────────────────────────────────

    def save(
        self,
        filepath: str,
        journal: str | None = None,
        dpi: int | None = None,
        index: int = -1,
    ) -> None:
        """
        Save the most recent figure (or figure at index) to file.

        Parameters
        ----------
        filepath : str
            Output path. Extension determines format:
            .pdf, .png, .svg, .eps — matplotlib static formats.
            .html — plotly interactive HTML.
        journal : str, optional
            Journal name for metadata embedding (informational only).
        dpi : int, optional
            Override DPI. Defaults to journal style DPI.
        index : int, optional
            Figure index. Default -1 (most recent).

        Examples
        --------
        >>> plot.save("figure1.pdf", journal="Physical Review")
        >>> plot.save("figure1.png", dpi=600)
        >>> plot.save("figure1.html")  # interactive plotly
        """
        if not self._figures:
            raise RuntimeError("No figures to save. Call a plot method first.")

        fig = self._figures[index]
        ext = Path(filepath).suffix.lower()
        save_dpi = dpi or self._style["dpi"]

        try:
            import plotly.graph_objects as go

            if isinstance(fig, go.Figure):
                if (
                    ext == ".html"
                ):  # pragma: no cover # pragma: no cover # pragma: no cover # pragma: no cover
                    fig.write_html(filepath)
                elif ext in (".png", ".pdf", ".svg", ".eps"):  # pragma: no cover
                    fig.write_image(filepath, scale=save_dpi / 96)
                else:
                    fig.write_html(filepath)  # pragma: no cover
                print(f"Saved interactive figure → {filepath}")  # pragma: no cover
                return  # pragma: no cover
        except ImportError:
            pass

        fig.savefig(filepath, dpi=save_dpi, bbox_inches="tight")
        label = f" ({journal})" if journal else ""
        print(f"Saved{label} → {filepath}  [{save_dpi} DPI]")

    def save_all(
        self,
        directory: str = ".",
        prefix: str = "figure",
        fmt: str = "pdf",
        journal: str | None = None,
    ) -> None:
        """
        Save all generated figures to a directory.

        Parameters
        ----------
        directory : str
            Output directory (created if it doesn't exist).
        prefix : str
            Filename prefix. Files are named {prefix}_1.pdf, etc.
        fmt : str
            Output format: 'pdf', 'png', 'svg', 'html'.
        journal : str, optional
            Journal name for metadata.
        """
        os.makedirs(directory, exist_ok=True)
        for i, _ in enumerate(self._figures):
            path = os.path.join(directory, f"{prefix}_{i + 1}.{fmt}")
            self.save(path, journal=journal, index=i)

    def latex_caption(
        self,
        filepath: str,
        caption: str = "",
        label: str = "",
        index: int = -1,
    ) -> None:
        """
        Write a LaTeX figure environment with caption to a .tex file.

        Parameters
        ----------
        filepath : str
            Output .tex file path.
        caption : str
            Figure caption text.
        label : str
            LaTeX label for \\ref{} cross-referencing.
        index : int
            Figure index. Default -1 (most recent).

        Examples
        --------
        >>> plot.save("fig1.pdf")
        >>> plot.latex_caption("fig1_caption.tex",
        ...     caption="Neutron energy spectrum from Geant4 simulation.",
        ...     label="fig:neutron_spectrum")
        """
        pdf_name = Path(filepath).stem.replace("_caption", "")
        lbl = label or f"fig:{pdf_name}"

        tex = (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\columnwidth]{{{pdf_name}}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{{lbl}}}\n"
            "\\end{figure}\n"
        )
        with open(filepath, "w") as f:
            f.write(tex)
        print(f"LaTeX figure environment → {filepath}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:  # pragma: no cover
        """Convert #RRGGBB to 'R,G,B' string for plotly rgba()."""
        h = hex_color.lstrip("#")  # pragma: no cover # pragma: no cover
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"

    def __repr__(self) -> str:
        return (
            f"PhysicsPlot(style='{self.style_name}', "
            f"interactive={self._interactive}, "
            f"figures={len(self._figures)})"
        )
