"""
tests/test_viz.py
─────────────────
Test suite for triples_sigfast.viz.PhysicsPlot

All tests use matplotlib backend (non-interactive) to avoid
requiring a display or Jupyter environment in CI.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from triples_sigfast.viz.physics_plot import _JOURNAL_STYLES, PhysicsPlot

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def spectrum_data():
    rng = np.random.default_rng(0)
    energies = np.linspace(0, 10, 200)
    counts = rng.poisson(500 * np.exp(-energies / 3) + 50).astype(float)
    smoothed = counts * 0.95
    peaks = np.array([20, 80, 140])
    errors = np.sqrt(counts)
    return energies, counts, smoothed, peaks, errors


@pytest.fixture
def mesh_data():
    x = np.linspace(0, 100, 30)
    y = np.linspace(0, 100, 30)
    X, Y = np.meshgrid(x, y)
    dose = np.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 500) * 100
    return x, y, dose.T


@pytest.fixture
def plot():
    """Non-interactive PhysicsPlot for all tests."""
    return PhysicsPlot(style="publication", interactive=False)


# ── Initialisation ────────────────────────────────────────────────────────────


class TestPhysicsPlotInit:
    def test_default_style(self):
        p = PhysicsPlot(interactive=False)
        assert p.style_name == "publication"

    def test_all_styles_instantiate(self):
        for style in PhysicsPlot.available_styles():
            p = PhysicsPlot(style=style, interactive=False)
            assert p.style_name == style

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown style"):
            PhysicsPlot(style="journal_of_unobtanium", interactive=False)

    def test_repr(self, plot):
        assert "PhysicsPlot" in repr(plot)
        assert "publication" in repr(plot)

    def test_available_styles_returns_list(self):
        styles = PhysicsPlot.available_styles()
        assert isinstance(styles, list)
        assert "physical_review" in styles
        assert "nature" in styles
        assert "jinst" in styles

    def test_custom_style_override(self):
        p = PhysicsPlot(
            style="publication",
            custom_style={"font_size": 99},
            interactive=False,
        )
        assert p._style["font_size"] == 99

    def test_register_custom_style(self):
        PhysicsPlot.register_style(
            "my_journal",
            {
                "fig_width": 7,
                "fig_height": 5,
                "font_size": 10,
                "font_family": "serif",
                "line_width": 1.5,
                "dpi": 300,
                "color_cycle": ["#000000"],
            },
        )
        assert "my_journal" in PhysicsPlot.available_styles()
        p = PhysicsPlot(style="my_journal", interactive=False)
        assert p.style_name == "my_journal"

    def test_register_partial_style_fills_defaults(self):
        PhysicsPlot.register_style("partial_style", {"font_size": 14})
        assert "partial_style" in _JOURNAL_STYLES
        assert "color_cycle" in _JOURNAL_STYLES["partial_style"]

    def test_interactive_false_when_plotly_not_requested(self):
        p = PhysicsPlot(interactive=False)
        assert p._interactive is False


# ── Spectrum ──────────────────────────────────────────────────────────────────


class TestSpectrum:
    def test_returns_figure(self, plot, spectrum_data):
        import matplotlib.figure

        energies, counts, smoothed, peaks, errors = spectrum_data
        fig = plot.spectrum(energies, counts)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_smoothed(self, plot, spectrum_data):
        import matplotlib.figure

        energies, counts, smoothed, peaks, errors = spectrum_data
        fig = plot.spectrum(energies, counts, smoothed=smoothed)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_peaks(self, plot, spectrum_data):
        import matplotlib.figure

        energies, counts, smoothed, peaks, errors = spectrum_data
        fig = plot.spectrum(energies, counts, smoothed=smoothed, peaks=peaks)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_errors(self, plot, spectrum_data):
        import matplotlib.figure

        energies, counts, smoothed, peaks, errors = spectrum_data
        fig = plot.spectrum(energies, counts, errors=errors)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_all_options(self, plot, spectrum_data):
        import matplotlib.figure

        energies, counts, smoothed, peaks, errors = spectrum_data
        fig = plot.spectrum(
            energies,
            counts,
            smoothed=smoothed,
            peaks=peaks,
            errors=errors,
            title="Test",
            xlabel="E (MeV)",
            ylabel="Counts",
            xscale="log",
            yscale="log",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_stored_in_figures_list(self, plot, spectrum_data):
        energies, counts, smoothed, peaks, errors = spectrum_data
        n_before = len(plot._figures)
        plot.spectrum(energies, counts)
        assert len(plot._figures) == n_before + 1

    def test_empty_peaks_array(self, plot, spectrum_data):
        import matplotlib.figure

        energies, counts, smoothed, peaks, errors = spectrum_data
        fig = plot.spectrum(energies, counts, peaks=np.array([], dtype=int))
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_all_journal_styles_produce_figure(self, spectrum_data):
        import matplotlib.figure

        energies, counts, *_ = spectrum_data
        for style in [
            "physical_review",
            "nature",
            "jinst",
            "nim_a",
            "softwarex",
            "presentation",
            "thesis",
        ]:
            p = PhysicsPlot(style=style, interactive=False)
            fig = p.spectrum(energies, counts)
            assert isinstance(fig, matplotlib.figure.Figure)


# ── Shielding comparison ──────────────────────────────────────────────────────


class TestShieldingComparison:
    def test_returns_figure(self, plot):
        import matplotlib.figure

        t = np.linspace(0, 20, 50)
        fig = plot.shielding_comparison(["lead", "concrete"], t, energy_mev=1.25)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_single_material(self, plot):
        import matplotlib.figure

        t = np.linspace(0, 15, 30)
        fig = plot.shielding_comparison(["water"], t)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_all_materials(self, plot):
        import matplotlib.figure

        from triples_sigfast.nuclear.shielding import available_materials

        t = np.linspace(0, 10, 20)
        fig = plot.shielding_comparison(available_materials(), t)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_buildup(self, plot):
        import matplotlib.figure

        t = np.linspace(0, 20, 30)
        fig = plot.shielding_comparison(["lead"], t, use_buildup=False)
        assert isinstance(fig, matplotlib.figure.Figure)


# ── Dose map ──────────────────────────────────────────────────────────────────


class TestDoseMap:
    def test_returns_figure(self, plot, mesh_data):
        import matplotlib.figure

        x, y, dose = mesh_data
        fig = plot.dose_map(x, y, dose)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_linear_scale(self, plot, mesh_data):
        import matplotlib.figure

        x, y, dose = mesh_data
        fig = plot.dose_map(x, y, dose, log_scale=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_colormap(self, plot, mesh_data):
        import matplotlib.figure

        x, y, dose = mesh_data
        fig = plot.dose_map(x, y, dose, colormap="plasma")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_labels(self, plot, mesh_data):
        import matplotlib.figure

        x, y, dose = mesh_data
        fig = plot.dose_map(
            x,
            y,
            dose,
            unit="mSv/hr",
            title="Dose Map",
            xlabel="X",
            ylabel="Y",
        )
        assert isinstance(fig, matplotlib.figure.Figure)


# ── Convergence map ───────────────────────────────────────────────────────────


class TestConvergenceMap:
    def test_returns_figure(self, plot):
        import matplotlib.figure

        R = np.linspace(0.01, 0.15, 50)
        fig = plot.convergence_map(R)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_energies(self, plot):
        import matplotlib.figure

        R = np.linspace(0.01, 0.15, 50)
        E = np.linspace(0, 10, 50)
        fig = plot.convergence_map(R, energies=E)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_threshold(self, plot):
        import matplotlib.figure

        R = np.linspace(0.01, 0.15, 30)
        fig = plot.convergence_map(R, threshold=0.10)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_all_converged(self, plot):
        import matplotlib.figure

        R = np.ones(20) * 0.01
        fig = plot.convergence_map(R, threshold=0.05)
        assert isinstance(fig, matplotlib.figure.Figure)


# ── Save / export ─────────────────────────────────────────────────────────────


class TestSaveExport:
    def test_save_pdf(self, plot, spectrum_data, tmp_path):
        energies, counts, *_ = spectrum_data
        plot.spectrum(energies, counts)
        path = str(tmp_path / "fig.pdf")
        plot.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_save_png(self, plot, spectrum_data, tmp_path):
        energies, counts, *_ = spectrum_data
        plot.spectrum(energies, counts)
        path = str(tmp_path / "fig.png")
        plot.save(path, dpi=72)
        assert os.path.exists(path)

    def test_save_svg(self, plot, spectrum_data, tmp_path):
        energies, counts, *_ = spectrum_data
        plot.spectrum(energies, counts)
        path = str(tmp_path / "fig.svg")
        plot.save(path)
        assert os.path.exists(path)

    def test_save_with_journal_name(self, plot, spectrum_data, tmp_path, capsys):
        energies, counts, *_ = spectrum_data
        plot.spectrum(energies, counts)
        path = str(tmp_path / "fig.pdf")
        plot.save(path, journal="Physical Review")
        out = capsys.readouterr().out
        assert "Physical Review" in out

    def test_save_no_figures_raises(self, tmp_path):
        p = PhysicsPlot(interactive=False)
        with pytest.raises(RuntimeError, match="No figures"):
            p.save(str(tmp_path / "fig.pdf"))

    def test_save_all(self, plot, spectrum_data, tmp_path):
        energies, counts, smoothed, peaks, errors = spectrum_data
        plot.spectrum(energies, counts)
        plot.spectrum(energies, smoothed)
        plot.save_all(directory=str(tmp_path), prefix="fig", fmt="png")
        files = list(tmp_path.glob("fig_*.png"))
        assert len(files) == 2

    def test_latex_caption(self, plot, spectrum_data, tmp_path):
        energies, counts, *_ = spectrum_data
        plot.spectrum(energies, counts)
        plot.save(str(tmp_path / "fig1.pdf"))
        tex_path = str(tmp_path / "fig1_caption.tex")
        plot.latex_caption(
            tex_path,
            caption="Neutron energy spectrum.",
            label="fig:neutron",
        )
        assert os.path.exists(tex_path)
        with open(tex_path) as f:
            content = f.read()
        assert "\\begin{figure}" in content
        assert "Neutron energy spectrum." in content
        assert "fig:neutron" in content
        assert "\\includegraphics" in content

    def test_save_by_index(self, plot, spectrum_data, tmp_path):
        energies, counts, smoothed, *_ = spectrum_data
        plot.spectrum(energies, counts)
        plot.spectrum(energies, smoothed)
        path = str(tmp_path / "first.pdf")
        plot.save(path, index=0)
        assert os.path.exists(path)


# ── Integration ───────────────────────────────────────────────────────────────


class TestVizIntegration:
    def test_full_pipeline_matplotlib(self, tmp_path):
        """Full research pipeline → save PDF → LaTeX caption."""
        import matplotlib.figure

        from triples_sigfast import find_peaks, savitzky_golay
        from triples_sigfast.stats.mc import relative_error

        rng = np.random.default_rng(1)
        energies = np.linspace(0, 10, 300)
        counts = rng.poisson(
            1000 * np.exp(-energies / 2) + 500 * np.exp(-((energies - 5) ** 2) / 0.1)
        ).astype(float)
        smoothed = savitzky_golay(counts, window=15, polyorder=3)
        peaks = find_peaks(smoothed, min_height=100, min_distance=20)
        R = relative_error(counts)

        plot = PhysicsPlot(style="physical_review", interactive=False)

        fig1 = plot.spectrum(
            energies,
            counts,
            smoothed=smoothed,
            peaks=peaks,
            title="Neutron Energy Spectrum",
            xlabel="Energy (MeV)",
            ylabel="Counts",
        )
        assert isinstance(fig1, matplotlib.figure.Figure)

        fig2 = plot.convergence_map(R, energies=energies)
        assert isinstance(fig2, matplotlib.figure.Figure)

        fig3 = plot.shielding_comparison(
            ["lead", "concrete", "polyethylene"],
            np.linspace(0, 20, 100),
        )
        assert isinstance(fig3, matplotlib.figure.Figure)

        # Save all
        plot.save_all(directory=str(tmp_path), prefix="figure", fmt="pdf")
        saved = list(tmp_path.glob("figure_*.pdf"))
        assert len(saved) == 3

        # LaTeX caption
        tex = str(tmp_path / "spectrum_caption.tex")
        plot.latex_caption(tex, caption="Test spectrum.", label="fig:test")
        assert os.path.exists(tex)
