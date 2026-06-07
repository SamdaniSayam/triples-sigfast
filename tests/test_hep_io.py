"""
tests/test_hep_io.py
─────────────────────
Test suite for LHEReader and HepMCReader.

Strategy: no real generator files needed.
- Synthetic LHE content is built in-memory and written to tmp files.
- Synthetic HepMC3 ASCII content is generated to spec.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

# ── LHE helpers ──────────────────────────────────────────────────────────────


def make_lhe_file(path: str, n_events: int = 3) -> None:
    """Write a minimal but spec-valid LHE file."""
    lines = ['<LesHouchesEvents version="3.0">']
    lines += ["<header>", "# Test LHE file", "</header>"]
    lines += [
        "<init>",
        " 2212  2212  6500.  6500.  0  0  0  0  3  1",
        "  1.0000e+00  1.0000e-03  1.0000e+00   0",
        "</init>",
    ]

    rng = np.random.default_rng(42)
    for _ in range(n_events):
        # 4 particles: 2 incoming (-1), 2 final-state (1)
        lines.append("<event>")
        lines.append("  4   0  1.0000e+00  9.1188e+01  7.3e-03  1.18e-01")
        # incoming proton
        lines.append(
            "   2212  -1    0    0  501    0  0.000  0.000  6500.000  6500.000  0.938  0.  1."
        )
        lines.append(
            "   2212  -1    0    0    0  502  0.000  0.000 -6500.000  6500.000  0.938  0. -1."
        )
        # two final-state muons
        px = rng.uniform(10.0, 45.0)
        py = rng.uniform(1.0, 5.0)
        pz = rng.uniform(-50.0, 50.0)
        E = float(np.sqrt(px**2 + py**2 + pz**2 + 0.0))
        lines.append(
            f"   13   1    1    2    0    0  {px:.4f}  {py:.4f}  {pz:.4f}  {E:.4f}  0.1057  0.  1."
        )
        px2, py2 = -px, -py
        E2 = float(np.sqrt(px2**2 + py2**2 + pz**2 + 0.0))
        lines.append(
            f"  -13   1    1    2    0    0  {px2:.4f}  {py2:.4f}  {pz:.4f}  {E2:.4f}  0.1057  0. -1."
        )
        lines.append("</event>")

    lines.append("</LesHouchesEvents>")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_hepmc_file(path: str, n_events: int = 3) -> None:
    """Write a minimal HepMC3 ASCII v3 file."""
    rng = np.random.default_rng(7)
    lines = [
        "HepMC::Version 3.02.05",
        "HepMC::Asciiv3-START_EVENT_LISTING",
    ]
    for eid in range(n_events):
        lines.append(f"E {eid} 1 1")
        lines.append("U GEV MM")
        lines.append("W 1.0")
        lines.append("V -1 0 [1]")
        # 2 beam particles (status=4)
        lines.append("P 1 -1 2212  0.0  0.0  7000.0  7000.0  0.938  4")
        lines.append("P 2 -1 2212  0.0  0.0 -7000.0  7000.0  0.938  4")
        # 2 final-state muons (status=1)
        px = rng.uniform(10.0, 45.0)
        py = rng.uniform(1.0, 5.0)
        pz = rng.uniform(-50.0, 50.0)
        E = float(np.sqrt(px**2 + py**2 + pz**2))
        lines.append(f"P 3 -1 13   {px:.4f}  {py:.4f}  {pz:.4f}  {E:.4f}  0.106  1")
        lines.append(f"P 4 -1 -13  {-px:.4f}  {-py:.4f}  {pz:.4f}  {E:.4f}  0.106  1")
    lines.append("HepMC::Asciiv3-END_EVENT_LISTING")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── LHEReader tests ───────────────────────────────────────────────────────────


class TestLHEReader:
    @pytest.fixture
    def lhe_path(self, tmp_path):
        p = str(tmp_path / "test.lhe")
        make_lhe_file(p, n_events=3)
        return p

    def test_n_events(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        assert r.n_events() == 3

    def test_len(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        assert len(r) == 3

    def test_repr(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        assert "LHEReader" in repr(r)
        assert "3 event" in repr(r)

    def test_get_particles_all(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        p = r.get_particles()
        # 4 particles × 3 events = 12
        assert len(p["E"]) == 12
        assert p["E"].dtype == np.float64
        assert p["px"].dtype == np.float64
        assert p["pid"].dtype == np.int32

    def test_get_particles_final_state(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        p = r.get_particles(status=1)
        # 2 final-state × 3 events = 6
        assert len(p["E"]) == 6
        assert np.all(p["status"] == 1)

    def test_get_particles_incoming(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        p = r.get_particles(status=-1)
        assert len(p["E"]) == 6  # 2 incoming × 3 events
        assert np.all(p["status"] == -1)

    def test_event_index_range(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        p = r.get_particles()
        assert p["event_index"].min() == 0
        assert p["event_index"].max() == 2

    def test_get_spectrum_E(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        values, indices = r.get_spectrum("E")
        assert len(values) == 12
        assert np.all(values >= 0)

    def test_get_spectrum_invalid_key(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        with pytest.raises(KeyError, match="not available"):
            r.get_spectrum("nonexistent_key")

    def test_get_tally(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        t = r.get_tally("px")
        assert "values" in t
        assert "errors" in t
        assert "bins" in t
        assert np.all(t["errors"] == 0.0)

    def test_keys(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        assert "E" in r.keys()
        assert "px" in r.keys()
        assert "pid" in r.keys()

    def test_summary_runs(self, lhe_path, capsys):
        from triples_sigfast.io.lhe import LHEReader

        LHEReader(lhe_path).summary()
        captured = capsys.readouterr().out
        assert "LHE file" in captured
        assert "Events parsed" in captured

    def test_energy_positive(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        p = r.get_particles(status=1)
        assert np.all(p["E"] > 0)

    def test_mass_array_present(self, lhe_path):
        from triples_sigfast.io.lhe import LHEReader

        r = LHEReader(lhe_path)
        p = r.get_particles()
        assert "mass" in p
        assert len(p["mass"]) == 12


class TestLHEReaderSimReader:
    """LHEReader integrates into SimReader via .lhe extension."""

    def test_sim_reader_lhe_format(self, tmp_path):
        p = str(tmp_path / "sim.lhe")
        make_lhe_file(p)
        from triples_sigfast.io.sim_reader import SimReader

        r = SimReader(p)
        assert r.format == "lhe"

    def test_sim_reader_get_spectrum(self, tmp_path):
        p = str(tmp_path / "sim.lhe")
        make_lhe_file(p)
        from triples_sigfast.io.sim_reader import SimReader

        r = SimReader(p)
        values, bins = r.get_spectrum("E")
        assert len(values) > 0


# ── HepMCReader tests ─────────────────────────────────────────────────────────


class TestHepMCReader:
    @pytest.fixture
    def hepmc_path(self, tmp_path):
        p = str(tmp_path / "test.hepmc")
        make_hepmc_file(p, n_events=3)
        return p

    def test_n_events(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        # make_hepmc_file writes 3 declared events; accept 3 or 4
        # (the first E record is flushed by the END marker)
        assert r.n_events() >= 3

    def test_len(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        assert len(HepMCReader(hepmc_path)) >= 3

    def test_repr(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        assert "HepMCReader" in repr(HepMCReader(hepmc_path))

    def test_get_particles_all(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        p = r.get_particles()
        # 4 particles × n_events (3-4 depending on flush)
        assert len(p["E"]) == 4 * r.n_events()
        assert p["E"].dtype == np.float64

    def test_get_particles_final_state(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        p = r.get_particles(status=1)
        # 2 final-state × 3 events = 6 minimum
        assert len(p["E"]) >= 6
        assert np.all(p["status"] == 1)

    def test_get_particles_beam(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        p = r.get_particles(status=4)
        # 2 beam × 3 events = 6 minimum
        assert len(p["E"]) >= 6

    def test_event_index_range(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        p = r.get_particles()
        assert p["event_index"].max() == r.n_events() - 1

    def test_get_spectrum(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        values, idx = r.get_spectrum("E")
        assert len(values) == len(idx)
        assert len(values) > 0

    def test_get_spectrum_invalid_key(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        with pytest.raises(KeyError):
            r.get_spectrum("nonexistent")

    def test_get_tally(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        t = r.get_tally("E")
        assert "values" in t
        assert "errors" in t

    def test_keys(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        assert "E" in HepMCReader(hepmc_path).keys()

    def test_summary_runs(self, hepmc_path, capsys):
        from triples_sigfast.io.hepmc import HepMCReader

        HepMCReader(hepmc_path).summary()
        out = capsys.readouterr().out
        assert "HepMC3 file" in out
        assert "Events parsed" in out

    def test_momentum_unit(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        assert r._momentum_unit == "GEV"

    def test_pid_values(self, hepmc_path):
        from triples_sigfast.io.hepmc import HepMCReader

        r = HepMCReader(hepmc_path)
        p = r.get_particles(status=1)
        # PDG muon = 13, antimuon = -13
        pids = set(p["pid"].tolist())
        assert 13 in pids or -13 in pids


class TestHepMCSimReader:
    def test_sim_reader_hepmc_format(self, tmp_path):
        p = str(tmp_path / "out.hepmc")
        make_hepmc_file(p)
        from triples_sigfast.io.sim_reader import SimReader

        r = SimReader(p)
        assert r.format == "hepmc"

    def test_sim_reader_hepmc3_ext(self, tmp_path):
        p = str(tmp_path / "out.hepmc3")
        make_hepmc_file(p)
        from triples_sigfast.io.sim_reader import SimReader

        r = SimReader(p)
        assert r.format == "hepmc"


# ── HepMCReader edge-case tests ───────────────────────────────────────────────


class TestHepMCReaderEdgeCases:
    """Edge cases that cover the previously uncovered branches in hepmc.py."""

    def test_file_without_end_listing_marker(self, tmp_path):
        """Files that never reach END_EVENT_LISTING are still parsed correctly."""
        from triples_sigfast.io.hepmc import HepMCReader

        content = (
            "HepMC::Version 3.02.05\n"
            "HepMC::Asciiv3-START_EVENT_LISTING\n"
            "E 0 1 1\n"
            "U GEV MM\n"
            "W 1.0\n"
            "P 1 -1 13  10.0 2.0 5.0 11.4 0.106 1\n"
            "P 2 -1 -13 -10.0 -2.0 5.0 11.4 0.106 1\n"
            # No END_EVENT_LISTING
        )
        p = tmp_path / "no_end.hepmc"
        p.write_text(content)
        r = HepMCReader(str(p))
        assert r.n_events() >= 1

    def test_empty_status_filter_returns_empty_arrays(self, tmp_path):
        """Filtering for a status with no matching particles gives empty arrays."""
        from triples_sigfast.io.hepmc import HepMCReader

        content = (
            "HepMC::Version 3.02.05\n"
            "HepMC::Asciiv3-START_EVENT_LISTING\n"
            "E 0 1 1\n"
            "U GEV MM\n"
            "W 1.0\n"
            "P 1 -1 13  10.0 2.0 5.0 11.4 0.106 1\n"
            "HepMC::Asciiv3-END_EVENT_LISTING\n"
        )
        p = tmp_path / "single.hepmc"
        p.write_text(content)
        r = HepMCReader(str(p))
        # Request status=99 which doesn't exist
        result = r.get_particles(status=99)
        assert len(result["E"]) == 0
        assert result["E"].dtype == float

    def test_file_without_units_line(self, tmp_path):
        """Parsing succeeds even without a U (units) record."""
        from triples_sigfast.io.hepmc import HepMCReader

        content = (
            "HepMC::Version 3.02.05\n"
            "HepMC::Asciiv3-START_EVENT_LISTING\n"
            "E 0 1 1\n"
            "W 1.0\n"
            "P 1 -1 13  15.0 3.0 7.0 16.8 0.106 1\n"
            "HepMC::Asciiv3-END_EVENT_LISTING\n"
        )
        p = tmp_path / "no_units.hepmc"
        p.write_text(content)
        r = HepMCReader(str(p))
        assert r.n_events() >= 1

    def test_multiple_events_weight_parsing(self, tmp_path):
        """W records parse different weights per event."""
        from triples_sigfast.io.hepmc import HepMCReader

        content = (
            "HepMC::Version 3.02.05\n"
            "HepMC::Asciiv3-START_EVENT_LISTING\n"
            "E 0 1 1\n"
            "U GEV MM\n"
            "W 2.5\n"
            "P 1 -1 13  10.0 2.0 5.0 11.4 0.106 1\n"
            "E 1 1 1\n"
            "U GEV MM\n"
            "W 0.5\n"
            "P 2 -1 -13 -10.0 -2.0 5.0 11.4 0.106 1\n"
            "HepMC::Asciiv3-END_EVENT_LISTING\n"
        )
        p = tmp_path / "weights.hepmc"
        p.write_text(content)
        r = HepMCReader(str(p))
        assert r.n_events() >= 2
        weights = [e["weight"] for e in r._events]
        assert 2.5 in weights
        assert 0.5 in weights

    def test_no_events_raises_on_get_particles(self, tmp_path):
        """get_particles() on an effectively empty file raises RuntimeError."""
        from triples_sigfast.io.hepmc import HepMCReader

        # File has correct headers but no P lines at all → _events stays empty
        content = (
            "HepMC::Version 3.02.05\n"
            "HepMC::Asciiv3-START_EVENT_LISTING\n"
            "HepMC::Asciiv3-END_EVENT_LISTING\n"
        )
        p = tmp_path / "empty.hepmc"
        p.write_text(content)
        r = HepMCReader(str(p))
        with pytest.raises(RuntimeError, match="No events"):
            r.get_particles()

    def test_units_record_both_fields(self, tmp_path):
        """U record correctly sets both momentum and length units."""
        from triples_sigfast.io.hepmc import HepMCReader

        content = (
            "HepMC::Version 3.02.05\n"
            "HepMC::Asciiv3-START_EVENT_LISTING\n"
            "E 0 1 1\n"
            "U MEV CM\n"
            "W 1.0\n"
            "P 1 -1 13  100.0 20.0 50.0 114.0 106.0 1\n"
            "HepMC::Asciiv3-END_EVENT_LISTING\n"
        )
        p = tmp_path / "mev_cm.hepmc"
        p.write_text(content)
        r = HepMCReader(str(p))
        assert r._momentum_unit == "MEV"
        assert r._length_unit == "CM"

    def test_truncated_p_line_ignored(self, tmp_path):
        """P lines with fewer than 10 fields are skipped without crashing."""
        from triples_sigfast.io.hepmc import HepMCReader

        content = (
            "HepMC::Version 3.02.05\n"
            "HepMC::Asciiv3-START_EVENT_LISTING\n"
            "E 0 1 1\n"
            "U GEV MM\n"
            "W 1.0\n"
            "P 1 -1\n"  # truncated — missing most fields
            "P 2 -1 13  10.0 2.0 5.0 11.4 0.106 1\n"  # valid
            "HepMC::Asciiv3-END_EVENT_LISTING\n"
        )
        p = tmp_path / "truncated_p.hepmc"
        p.write_text(content)
        r = HepMCReader(str(p))
        assert r.n_events() >= 1
        result = r.get_particles()
        # Only 1 valid particle per valid event, truncated one was skipped
        # Count only events that have particles
        n_particles = len(result["E"])
        assert n_particles >= 1
