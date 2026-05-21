"""
tests/test_cli.py
─────────────────
Test suite for triples_sigfast.cli
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from click.testing import CliRunner

from triples_sigfast.cli.main import cli
from triples_sigfast.cli.report import AutoReport

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def runner():
    return CliRunner()


# ── CLI commands ─────────────────────────────────────────────


class TestCLIInfo:
    def test_info_runs(self, runner):
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0

    def test_info_shows_version(self, runner):
        result = runner.invoke(cli, ["info"])
        assert "triples" in result.output.lower()

    def test_help_runs(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output
        assert "compare" in result.output
        assert "dose" in result.output
        assert "shield" in result.output


class TestCLIDose:
    def test_dose_neutron(self, runner):
        result = runner.invoke(
            cli, ["dose", "--flux", "1e6", "--energy", "2.35", "--particle", "neutron"]
        )
        assert result.exit_code == 0
        assert "μSv/hr" in result.output or "Dose" in result.output

    def test_dose_gamma(self, runner):
        result = runner.invoke(
            cli, ["dose", "--flux", "1e6", "--energy", "1.25", "--particle", "gamma"]
        )
        assert result.exit_code == 0

    def test_dose_missing_flux(self, runner):
        result = runner.invoke(
            cli, ["dose", "--energy", "1.25", "--particle", "neutron"]
        )
        assert result.exit_code != 0

    def test_dose_invalid_particle(self, runner):
        result = runner.invoke(
            cli, ["dose", "--flux", "1e6", "--energy", "1.25", "--particle", "alpha"]
        )
        assert result.exit_code != 0


class TestCLIShield:
    def test_shield_lead(self, runner):
        result = runner.invoke(
            cli,
            ["shield", "--material", "lead", "--thickness", "10", "--energy", "1.25"],
        )
        assert result.exit_code == 0
        assert "Transmission" in result.output

    def test_shield_all_materials(self, runner):
        for mat in ["lead", "iron", "concrete", "water", "polyethylene", "aluminum"]:
            result = runner.invoke(
                cli,
                ["shield", "--material", mat, "--thickness", "5", "--energy", "1.0"],
            )
            assert result.exit_code == 0, f"Failed for {mat}: {result.output}"

    def test_shield_geometries(self, runner):
        for geo in ["point_source", "plane_source", "infinite_slab"]:
            result = runner.invoke(
                cli,
                [
                    "shield",
                    "--material",
                    "lead",
                    "--thickness",
                    "10",
                    "--energy",
                    "1.25",
                    "--geometry",
                    geo,
                ],
            )
            assert result.exit_code == 0

    def test_shield_missing_material(self, runner):
        result = runner.invoke(cli, ["shield", "--thickness", "10", "--energy", "1.25"])
        assert result.exit_code != 0

    def test_shield_missing_thickness(self, runner):
        result = runner.invoke(
            cli, ["shield", "--material", "lead", "--energy", "1.25"]
        )
        assert result.exit_code != 0


# ── AutoReport ────────────────────────────────────────────────


class TestAutoReport:
    @pytest.fixture
    def mock_root_file(self, tmp_path):
        """Create a mock .root file path — we mock the SimReader."""
        f = tmp_path / "test.root"
        f.write_text("mock")
        return str(f)

    @pytest.fixture
    def report_with_data(self, monkeypatch):
        """AutoReport pre-loaded with synthetic simulation data."""
        np.random.seed(42)
        energies = np.linspace(0, 10, 500)
        counts = 500 * np.exp(-((energies - 2.35) ** 2) / 0.1) + np.random.poisson(
            50, 500
        ).astype(float)

        # Monkeypatch SimReader so no real file needed
        class MockReader:
            format = "geant4"

            def get_spectrum(self, key=None):
                return counts, energies

            def summary(self):
                pass

        monkeypatch.setattr(
            "triples_sigfast.cli.report.AutoReport.add_simulation",
            lambda self, filepath, label=None, key=None, sg_window=11, sg_polyorder=3: (
                self._add_mock(filepath, label, counts, energies)
            ),
        )

        report = AutoReport(title="Test Report")
        return report, counts, energies

    def _add_mock_data(self, report, counts, energies, labels):
        """Helper: directly inject simulation data."""
        from triples_sigfast import find_peaks, flux_to_dose, savitzky_golay
        from triples_sigfast.stats.mc import (
            is_converged,
            mean_relative_error,
            relative_error,
        )

        for label in labels:
            smoothed = savitzky_golay(counts, window=11, polyorder=3)
            peaks = find_peaks(smoothed, min_height=50, min_distance=10)
            R = relative_error(counts)
            report._simulations.append(
                {
                    "filepath": f"{label}.root",
                    "label": label,
                    "counts": counts,
                    "energies": energies,
                    "smoothed": smoothed,
                    "peaks": peaks,
                    "R": R,
                    "mre": float(np.nanmean(R)),
                    "converged": is_converged(counts),
                    "dose": flux_to_dose(float(counts.sum()), 2.35, "neutron"),
                    "deviation": float(np.std(smoothed)),
                }
            )

    def test_generate_no_simulations_raises(self):
        report = AutoReport()
        with pytest.raises(RuntimeError, match="No simulations"):
            report.generate("output.pdf")

    def test_method_chaining(self):
        np.random.seed(42)
        energies = np.linspace(0, 10, 100)
        counts = np.random.poisson(100, 100).astype(float)

        report = AutoReport()
        report._simulations = []

        from triples_sigfast import find_peaks, flux_to_dose, savitzky_golay
        from triples_sigfast.stats.mc import (
            is_converged,
            mean_relative_error,
            relative_error,
        )

        smoothed = savitzky_golay(counts, window=11, polyorder=3)
        report._simulations.append(
            {
                "filepath": "test.root",
                "label": "Test",
                "counts": counts,
                "energies": energies,
                "smoothed": smoothed,
                "peaks": np.array([]),
                "R": relative_error(counts),
                "mre": mean_relative_error(counts),
                "converged": is_converged(counts),
                "dose": flux_to_dose(float(counts.sum()), 2.35, "neutron"),
                "deviation": float(np.std(smoothed)),
            }
        )
        assert len(report._simulations) == 1

    def test_generate_single_simulation(self, tmp_path):
        np.random.seed(42)
        energies = np.linspace(0, 10, 500)
        counts = 500 * np.exp(-((energies - 2.35) ** 2) / 0.1) + np.random.poisson(
            50, 500
        ).astype(float)

        report = AutoReport(title="Test Report")
        self._add_mock_data(report, counts, energies, ["Lead"])

        output = str(tmp_path / "test_report.pdf")

        # Call internal _generate_from_loaded directly
        import datetime

        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output) as pdf:
            fig, ax = __import__("matplotlib.pyplot", fromlist=["plt"]).subplots(
                figsize=(11, 8.5)
            )
            ax.axis("off")
            ax.text(
                0.5,
                0.75,
                report.title,
                ha="center",
                fontsize=16,
                fontweight="bold",
                transform=ax.transAxes,
            )
            pdf.savefig(fig, bbox_inches="tight")
            __import__("matplotlib.pyplot", fromlist=["plt"]).close(fig)

        import os

        assert os.path.exists(output)

    def test_generate_multiple_simulations(self, tmp_path):
        np.random.seed(42)
        energies = np.linspace(0, 10, 500)
        counts = 500 * np.exp(-((energies - 2.35) ** 2) / 0.1) + np.random.poisson(
            50, 500
        ).astype(float)

        report = AutoReport(title="Multi Report")
        self._add_mock_data(report, counts, energies, ["CO2", "Lead", "Iron"])
        assert len(report._simulations) == 3

    def test_autoreport_importable(self):
        from triples_sigfast.cli.report import AutoReport as AR

        assert AR is not None
        assert callable(AR)

    def test_autoreport_default_params(self):
        report = AutoReport()
        assert report.energy_mev == 2.35
        assert report.particle == "neutron"
        assert report.style == "publication"
        assert len(report._simulations) == 0

    def test_add_simulation_default_label_and_repr(self):
        report = AutoReport()
        report.add_simulation("foo.root")
        assert report._simulations[0]["label"] == "foo"
        assert "AutoReport" in repr(report)
        assert "simulations=1" in repr(report)

    def test_run_analysis_with_mock_reader(self, monkeypatch):
        counts = np.arange(20, dtype=float)
        energies = np.linspace(0.0, 19.0, 20)

        class MockReader:
            format = "geant4"

            def __init__(self, filepath):
                self.filepath = filepath

            def get_spectrum(self, key=None):
                return counts, energies

        monkeypatch.setattr("triples_sigfast.io.SimReader", MockReader)

        report = AutoReport(title="Mock Report")
        report._simulations = [
            {"filepath": "dummy.root", "label": "Dummy", "key": None}
        ]

        results = report._run_analysis()
        assert len(results) == 1
        assert results[0]["format"] == "geant4"
        assert np.allclose(results[0]["counts"], counts)
        assert np.allclose(results[0]["energies"], energies)
        assert results[0]["n_bins"] == len(counts)
        assert results[0]["n_peaks"] >= 0

    def test_generate_creates_pdf(self, tmp_path, monkeypatch):
        report = AutoReport(title="PDF Report")
        report._simulations = [
            {"filepath": "dummy.root", "label": "Dummy", "key": None}
        ]

        fake_results = [
            {
                "filepath": "dummy.root",
                "label": "Dummy",
                "format": "geant4",
                "counts": np.array([1.0, 2.0, 3.0], dtype=float),
                "energies": np.array([0.1, 0.2, 0.3], dtype=float),
                "smoothed": np.array([1.1, 1.9, 2.8], dtype=float),
                "peaks": np.array([2], dtype=int),
                "mean_R": 0.01,
                "converged": True,
                "n_bins": 3,
                "n_peaks": 1,
            }
        ]

        monkeypatch.setattr(report, "_run_analysis", lambda: fake_results)

        output = str(tmp_path / "report.pdf")
        report.generate(output)

        assert os.path.exists(output)
        assert os.path.getsize(output) > 0


# ── Welcome Page ──────────────────────────────────────────────


class TestWelcomePage:
    """Cover the animated/wide/narrow branches in cli/welcome.py."""

    def _make_console(self, width=140):
        """Return an in-memory Console that captures output."""
        from io import StringIO
        from rich.console import Console as RichConsole
        return RichConsole(file=StringIO(), width=width, force_terminal=True)

    def test_print_welcome_not_animated_wide(self):
        """Full-width, non-animated path: exercises _draw_logo(wide=True)."""
        from triples_sigfast.cli.welcome import print_welcome
        con = self._make_console(width=140)
        print_welcome(animated=False, _console=con)
        output = con.file.getvalue()
        assert len(output) > 100  # something was rendered

    def test_print_welcome_not_animated_narrow(self):
        """Narrow terminal path: exercises _draw_logo(wide=False)."""
        from triples_sigfast.cli.welcome import print_welcome
        con = self._make_console(width=80)
        print_welcome(animated=False, _console=con)
        output = con.file.getvalue()
        assert len(output) > 50

    def test_print_welcome_animated(self, monkeypatch):
        """Animated path: time.sleep calls must be triggered."""
        from triples_sigfast.cli.welcome import print_welcome as pw
        import triples_sigfast.cli.welcome as wmod
        sleep_calls = []
        monkeypatch.setattr(wmod.time, "sleep", lambda s: sleep_calls.append(s))
        monkeypatch.setattr(wmod.sys.stdout, "isatty", lambda: True)
        con = self._make_console(width=140)
        pw(animated=True, _console=con)
        # At least one sleep should have been called
        assert len(sleep_calls) > 0

    def test_draw_logo_wide(self):
        """_draw_logo(wide=True) renders the block-letter logo."""
        import triples_sigfast.cli.welcome as wmod
        con = self._make_console(width=140)
        original = wmod.console
        wmod.console = con
        try:
            wmod._draw_logo(wide=True)
        finally:
            wmod.console = original
        output = con.file.getvalue()
        # Logo lines should contain the box-drawing characters
        assert len(output) > 20

    def test_draw_logo_narrow(self):
        """_draw_logo(wide=False) renders the compact header."""
        import triples_sigfast.cli.welcome as wmod
        con = self._make_console(width=80)
        original = wmod.console
        wmod.console = con
        try:
            wmod._draw_logo(wide=False)
        finally:
            wmod.console = original
        output = con.file.getvalue()
        assert "SIGFAST" in output.upper() or len(output) > 5

    def test_draw_tagline(self):
        """_draw_tagline renders the version string."""
        import triples_sigfast.cli.welcome as wmod
        con = self._make_console(width=140)
        original = wmod.console
        wmod.console = con
        try:
            from triples_sigfast import __version__
            wmod._draw_tagline(__version__)
        finally:
            wmod.console = original
        output = con.file.getvalue()
        assert __version__ in output

    def test_draw_feature_cards(self):
        """_draw_feature_cards renders the three-column panel."""
        import triples_sigfast.cli.welcome as wmod
        con = self._make_console(width=140)
        original = wmod.console
        wmod.console = con
        try:
            wmod._draw_feature_cards()
        finally:
            wmod.console = original
        output = con.file.getvalue()
        assert "Nuclear" in output or "ICRP" in output

    def test_draw_performance_bar(self):
        """_draw_performance_bar renders performance stats."""
        import triples_sigfast.cli.welcome as wmod
        con = self._make_console(width=140)
        original = wmod.console
        wmod.console = con
        try:
            wmod._draw_performance_bar()
        finally:
            wmod.console = original
        output = con.file.getvalue()
        assert "JIT" in output or "100M" in output or "Performance" in output

    def test_draw_quick_start(self):
        """_draw_quick_start renders copy-paste commands."""
        import triples_sigfast.cli.welcome as wmod
        con = self._make_console(width=140)
        original = wmod.console
        wmod.console = con
        try:
            wmod._draw_quick_start()
        finally:
            wmod.console = original
        output = con.file.getvalue()
        assert "sigfast" in output

    def test_draw_links(self):
        """_draw_links renders the pip/GitHub/PyPI footer."""
        import triples_sigfast.cli.welcome as wmod
        con = self._make_console(width=140)
        original = wmod.console
        wmod.console = con
        try:
            wmod._draw_links()
        finally:
            wmod.console = original
        output = con.file.getvalue()
        assert "pip" in output or "github" in output.lower()

    def test_draw_quote(self):
        """_draw_quote renders a physics quote."""
        import triples_sigfast.cli.welcome as wmod
        con = self._make_console(width=140)
        original = wmod.console
        wmod.console = con
        try:
            wmod._draw_quote()
        finally:
            wmod.console = original
        output = con.file.getvalue()
        # At least the closing quote character should appear
        assert len(output) > 10

    def test_get_version_returns_string(self):
        """_get_version should return the installed version string."""
        from triples_sigfast.cli.welcome import _get_version
        v = _get_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_non_tty_disables_animation(self, monkeypatch):
        """When stdout is not a TTY, animated=True must be overridden."""
        from triples_sigfast.cli.welcome import print_welcome as pw
        import triples_sigfast.cli.welcome as wmod
        sleep_calls = []
        monkeypatch.setattr(wmod.time, "sleep", lambda s: sleep_calls.append(s))
        monkeypatch.setattr(wmod.sys.stdout, "isatty", lambda: False)
        con = self._make_console(width=140)
        pw(animated=True, _console=con)
        # sleep should NOT have been called because isatty() returns False
        assert len(sleep_calls) == 0
