"""
tests/test_readers.py
─────────────────────
Test suite for triples_sigfast.io native readers:
FlukaReader, MCNPReader, SerpentReader.

All tests use synthetic files that mirror real simulation output formats.
No actual simulation codes are required to run these tests.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from triples_sigfast.io.fluka import FlukaReader
from triples_sigfast.io.mcnp import MCNPReader
from triples_sigfast.io.serpent import SerpentReader

# ═══════════════════════════════════════════════════════════════════
# File factories
# ═══════════════════════════════════════════════════════════════════


def make_fluka_file(
    path: str,
    detectors: dict | None = None,
) -> None:
    """Create a synthetic FLUKA-style ASCII output file."""
    if detectors is None:
        detectors = {
            "neutron_fluence": (
                np.linspace(0.1, 10.0, 20),
                np.abs(np.random.default_rng(0).standard_normal(20)) * 100,
                np.ones(20) * 0.05,
            ),
            "gamma_dose": (
                np.linspace(0.01, 5.0, 15),
                np.abs(np.random.default_rng(1).standard_normal(15)) * 50,
                None,
            ),
        }

    lines = []
    for det_name, data in detectors.items():
        energies, values, errors = data
        lines.append(f"# DETECTOR: {det_name}")
        lines.append("# ESTIMATOR: USRTRACK")
        for i, (e, v) in enumerate(zip(energies, values)):
            if errors is not None:
                lines.append(f"{e:.6E}  {v:.6E}  {errors[i]:.6E}")
            else:
                lines.append(f"{e:.6E}  {v:.6E}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_mctal_file(path: str, n_tallies: int = 2) -> None:
    """Create a synthetic MCNP MCTAL file."""
    lines = [
        "mcnp   6    triples_sigfast_test_problem",
        f"ntal  {n_tallies}",
        "",
    ]

    tally_configs = [
        (
            "4",
            "n",
            [0.1, 0.5, 1.0, 2.0, 5.0],
            [1.23e-4, 4.56e-4, 7.89e-4, 2.11e-3, 5.00e-4],
        ),
        ("14", "p", [0.5, 1.0, 2.0, 5.0], [3.21e-5, 6.54e-5, 9.87e-5, 1.11e-4]),
    ]

    for i, (num, particle, energies, values) in enumerate(tally_configs[:n_tallies]):
        lines.append(f"tally {num}")
        lines.append(f"f{num}:{particle}  1")
        lines.append("")

        # Energy bins
        e_str = "  ".join(f"{e:.4E}" for e in energies)
        lines.append(f"et  {e_str}")
        lines.append("")

        # Values + errors
        lines.append("vals")
        for v in values:
            lines.append(f"  {v:.6E}  {0.05:.6E}")
        lines.append("")

        # TFC block (nps, mean, error, fom)
        lines.append("tfc  1")
        for j in range(1, 5):
            nps = j * 1000000
            mean = values[-1] * (1 + 0.01 * j)
            err = 0.10 / np.sqrt(j)
            fom = 1.0 / (err**2 * nps / 1e6)
            lines.append(f"  {nps}  {mean:.6E}  {err:.6E}  {fom:.6E}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_serpent_file(
    path: str,
    n_bins: int = 5,
    include_keff: bool = True,
    include_burnup: bool = False,
) -> None:
    """Create a synthetic SERPENT2 detector output file."""
    lines = []

    # Detector array (N * 12 columns)
    rows = []
    rng = np.random.default_rng(42)
    for i in range(n_bins):
        e_low = float(i)
        e_high = float(i + 1)
        flux = rng.uniform(1e-4, 1e-3)
        err = rng.uniform(0.01, 0.05)
        row = [e_low, e_high] + [0.0] * 8 + [flux, err]
        rows.append("  ".join(f"{v:.6E}" for v in row))

    block = "\n".join(rows)
    lines.append(f"DET_NEUTRON_FLUX = [\n{block}\n];")
    lines.append("")

    # Second detector
    rows2 = []
    for i in range(n_bins):
        e_low = float(i) * 0.5
        e_high = float(i + 1) * 0.5
        flux = rng.uniform(1e-5, 1e-4)
        err = rng.uniform(0.02, 0.08)
        row = [e_low, e_high] + [0.0] * 8 + [flux, err]
        rows2.append("  ".join(f"{v:.6E}" for v in row))

    block2 = "\n".join(rows2)
    lines.append(f"DET_GAMMA_FLUX = [\n{block2}\n];")
    lines.append("")

    if include_keff:
        lines.append("ANA_KEFF = [  1.023456E+00  2.345678E-04 ];")
        lines.append("IMP_KEFF = [  1.023123E+00  1.987654E-04 ];")
        lines.append("")

    if include_burnup:
        lines.append("BURNUP = [  0.0  1.0  5.0  10.0 ];")
        lines.append("BURN_DAYS = [  0.0  10.0  50.0  100.0 ];")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# FlukaReader
# ═══════════════════════════════════════════════════════════════════


class TestFlukaReader:
    @pytest.fixture
    def fluka_file(self, tmp_path):
        path = str(tmp_path / "sim.flair")
        make_fluka_file(path)
        return path

    @pytest.fixture
    def fluka_file_no_errors(self, tmp_path):
        path = str(tmp_path / "sim_noerr.flair")
        make_fluka_file(
            path,
            detectors={"flux": (np.linspace(0.1, 5, 10), np.ones(10) * 1e-3, None)},
        )
        return path

    def test_repr(self, fluka_file):
        r = FlukaReader(fluka_file)
        assert "FlukaReader" in repr(r)

    def test_len(self, fluka_file):
        r = FlukaReader(fluka_file)
        assert len(r) == 2

    def test_keys_returns_list(self, fluka_file):
        r = FlukaReader(fluka_file)
        keys = r.keys()
        assert isinstance(keys, list)
        assert "neutron_fluence" in keys
        assert "gamma_dose" in keys

    def test_get_spectrum_default(self, fluka_file):
        r = FlukaReader(fluka_file)
        values, bins = r.get_spectrum()
        assert len(values) == 20
        assert len(bins) == 20

    def test_get_spectrum_named(self, fluka_file):
        r = FlukaReader(fluka_file)
        values, bins = r.get_spectrum("neutron_fluence")
        assert len(values) == 20
        assert np.all(values >= 0)

    def test_get_spectrum_partial_match(self, fluka_file):
        r = FlukaReader(fluka_file)
        values, bins = r.get_spectrum("gamma")
        assert len(values) == 15

    def test_get_usrbin(self, fluka_file):
        r = FlukaReader(fluka_file)
        det = r.get_usrbin("neutron_fluence")
        assert "values" in det
        assert "bins" in det
        assert "estimator" in det

    def test_get_usrbdx(self, fluka_file):
        r = FlukaReader(fluka_file)
        det = r.get_usrbdx("neutron_fluence")
        assert "values" in det

    def test_get_tally(self, fluka_file):
        r = FlukaReader(fluka_file)
        det = r.get_tally("neutron_fluence")
        assert det["name"] == "neutron_fluence"

    def test_errors_parsed_correctly(self, fluka_file):
        r = FlukaReader(fluka_file)
        det = r.get_tally("neutron_fluence")
        assert np.all(det["errors"] >= 0)
        assert len(det["errors"]) == len(det["values"])

    def test_no_errors_column_gives_zero_errors(self, fluka_file_no_errors):
        r = FlukaReader(fluka_file_no_errors)
        det = r.get_tally("flux")
        assert np.all(det["errors"] == 0)

    def test_missing_detector_raises(self, fluka_file):
        r = FlukaReader(fluka_file)
        with pytest.raises(KeyError):
            r.get_tally("nonexistent_detector_xyz")

    def test_summary_runs(self, fluka_file, capsys):
        FlukaReader(fluka_file).summary()
        out = capsys.readouterr().out
        assert "FLUKA" in out
        assert "neutron_fluence" in out

    def test_bins_match_input_energies(self, fluka_file):
        r = FlukaReader(fluka_file)
        _, bins = r.get_spectrum("neutron_fluence")
        assert abs(bins[0] - 0.1) < 0.01
        assert abs(bins[-1] - 10.0) < 0.1

    def test_estimator_tag_parsed(self, fluka_file):
        r = FlukaReader(fluka_file)
        det = r.get_tally("neutron_fluence")
        assert det["estimator"] == "USRTRACK"

    def test_empty_file_gives_no_detectors(self, tmp_path):
        path = str(tmp_path / "empty.flair")
        with open(path, "w") as f:
            f.write("")
        r = FlukaReader(path)
        assert len(r) == 0

    def test_get_spectrum_empty_raises(self, tmp_path):
        path = str(tmp_path / "empty.flair")
        with open(path, "w") as f:
            f.write("")
        r = FlukaReader(path)
        with pytest.raises(RuntimeError):
            r.get_spectrum()


# ═══════════════════════════════════════════════════════════════════
# MCNPReader
# ═══════════════════════════════════════════════════════════════════


class TestMCNPReader:
    @pytest.fixture
    def mctal_file(self, tmp_path):
        path = str(tmp_path / "sim.mctal")
        make_mctal_file(path, n_tallies=2)
        return path

    @pytest.fixture
    def mctal_single(self, tmp_path):
        path = str(tmp_path / "single.mctal")
        make_mctal_file(path, n_tallies=1)
        return path

    def test_repr(self, mctal_file):
        r = MCNPReader(mctal_file)
        assert "MCNPReader" in repr(r)

    def test_len(self, mctal_file):
        r = MCNPReader(mctal_file)
        assert len(r) == 2

    def test_keys_returns_list(self, mctal_file):
        r = MCNPReader(mctal_file)
        keys = r.keys()
        assert isinstance(keys, list)
        assert "tally_4" in keys

    def test_tally_numbers(self, mctal_file):
        r = MCNPReader(mctal_file)
        nums = r.tally_numbers()
        assert "4" in nums

    def test_get_tally_by_int(self, mctal_file):
        r = MCNPReader(mctal_file)
        t = r.get_tally(4)
        assert "values" in t
        assert "errors" in t
        assert "bins" in t

    def test_get_tally_by_string(self, mctal_file):
        r = MCNPReader(mctal_file)
        t = r.get_tally("tally_4")
        assert t["tally_number"] == "4"

    def test_get_tally_partial_match(self, mctal_file):
        r = MCNPReader(mctal_file)
        t = r.get_tally("4")
        assert "values" in t

    def test_get_tally_missing_raises(self, mctal_file):
        r = MCNPReader(mctal_file)
        with pytest.raises(KeyError):
            r.get_tally(999)

    def test_values_positive(self, mctal_file):
        r = MCNPReader(mctal_file)
        t = r.get_tally(4)
        assert np.all(t["values"] > 0)

    def test_errors_in_valid_range(self, mctal_file):
        r = MCNPReader(mctal_file)
        t = r.get_tally(4)
        assert np.all(t["errors"] >= 0)
        assert np.all(t["errors"] <= 1.0)

    def test_bins_positive(self, mctal_file):
        r = MCNPReader(mctal_file)
        t = r.get_tally(4)
        assert np.all(t["bins"] >= 0)

    def test_get_spectrum_default(self, mctal_file):
        r = MCNPReader(mctal_file)
        values, bins = r.get_spectrum()
        assert len(values) > 0
        assert len(values) == len(bins)

    def test_get_spectrum_named(self, mctal_file):
        r = MCNPReader(mctal_file)
        values, bins = r.get_spectrum("tally_4")
        assert len(values) > 0

    def test_get_spectrum_empty_raises(self, tmp_path):
        path = str(tmp_path / "empty.mctal")
        with open(path, "w") as f:
            f.write("mcnp 6 empty\nntal 0\n")
        r = MCNPReader(path)
        with pytest.raises(RuntimeError):
            r.get_spectrum()

    def test_get_fom_returns_array(self, mctal_file):
        r = MCNPReader(mctal_file)
        fom = r.get_fom()
        assert isinstance(fom, np.ndarray)
        assert len(fom) > 0

    def test_fom_positive(self, mctal_file):
        r = MCNPReader(mctal_file)
        fom = r.get_fom()
        assert np.all(fom > 0)

    def test_get_fom_no_tfc_returns_empty(self, tmp_path):
        path = str(tmp_path / "notfc.mctal")
        lines = [
            "mcnp 6 notfc",
            "ntal 1",
            "",
            "tally 4",
            "f4:n 1",
            "",
            "et  0.1  1.0  10.0",
            "",
            "vals",
            "  1.23E-04  0.05",
            "  4.56E-04  0.03",
            "",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        r = MCNPReader(path)
        fom = r.get_fom()
        assert isinstance(fom, np.ndarray)

    def test_summary_runs(self, mctal_file, capsys):
        MCNPReader(mctal_file).summary()
        out = capsys.readouterr().out
        assert "MCNP" in out

    def test_header_parsed(self, mctal_file):
        r = MCNPReader(mctal_file)
        assert r._header.get("code") == "mcnp"

    def test_particle_type_stored(self, mctal_file):
        r = MCNPReader(mctal_file)
        t = r.get_tally(4)
        assert t["particle"] == "n"

    def test_multiple_tallies_independent(self, mctal_file):
        r = MCNPReader(mctal_file)
        t4 = r.get_tally(4)
        t14 = r.get_tally(14)
        assert not np.array_equal(t4["values"], t14["values"])


# ═══════════════════════════════════════════════════════════════════
# SerpentReader
# ═══════════════════════════════════════════════════════════════════


class TestSerpentReader:
    @pytest.fixture
    def serpent_file(self, tmp_path):
        path = str(tmp_path / "sim.det")
        make_serpent_file(path, n_bins=5, include_keff=True)
        return path

    @pytest.fixture
    def serpent_burnup(self, tmp_path):
        path = str(tmp_path / "burnup.det")
        make_serpent_file(path, n_bins=3, include_keff=True, include_burnup=True)
        return path

    def test_repr(self, serpent_file):
        r = SerpentReader(serpent_file)
        assert "SerpentReader" in repr(r)

    def test_len(self, serpent_file):
        r = SerpentReader(serpent_file)
        assert len(r) == 2

    def test_keys_returns_list(self, serpent_file):
        r = SerpentReader(serpent_file)
        keys = r.keys()
        assert isinstance(keys, list)
        assert "DET_NEUTRON_FLUX" in keys
        assert "DET_GAMMA_FLUX" in keys

    def test_get_detector_default(self, serpent_file):
        r = SerpentReader(serpent_file)
        values, bins = r.get_detector()
        assert len(values) == 5
        assert len(bins) == 5

    def test_get_detector_named(self, serpent_file):
        r = SerpentReader(serpent_file)
        values, bins = r.get_detector("DET_NEUTRON_FLUX")
        assert len(values) == 5
        assert np.all(values > 0)

    def test_get_detector_partial_match(self, serpent_file):
        r = SerpentReader(serpent_file)
        values, bins = r.get_detector("NEUTRON")
        assert len(values) == 5

    def test_get_spectrum_alias(self, serpent_file):
        r = SerpentReader(serpent_file)
        values, bins = r.get_spectrum()
        assert len(values) == 5

    def test_get_tally_full_dict(self, serpent_file):
        r = SerpentReader(serpent_file)
        det = r.get_tally("DET_NEUTRON_FLUX")
        assert "raw" in det
        assert "e_low" in det
        assert "e_high" in det
        assert "errors" in det

    def test_bin_centres_correct(self, serpent_file):
        r = SerpentReader(serpent_file)
        _, bins = r.get_detector("DET_NEUTRON_FLUX")
        np.testing.assert_allclose(bins[0], 0.5, rtol=1e-6)
        np.testing.assert_allclose(bins[1], 1.5, rtol=1e-6)

    def test_errors_in_range(self, serpent_file):
        r = SerpentReader(serpent_file)
        det = r.get_tally("DET_NEUTRON_FLUX")
        assert np.all(det["errors"] >= 0)
        assert np.all(det["errors"] <= 1.0)

    def test_get_keff(self, serpent_file):
        r = SerpentReader(serpent_file)
        keff = r.get_keff()
        assert abs(keff["ana_keff"] - 1.023456) < 1e-4
        assert keff["ana_err"] > 0

    def test_get_keff_no_data_returns_nan(self, tmp_path):
        path = str(tmp_path / "nokeff.det")
        make_serpent_file(path, include_keff=False)
        r = SerpentReader(path)
        keff = r.get_keff()
        assert np.isnan(keff["ana_keff"])

    def test_get_burnup(self, serpent_burnup):
        r = SerpentReader(serpent_burnup)
        burnup = r.get_burnup()
        assert len(burnup["burnup_MWd_kgU"]) == 4
        assert len(burnup["days"]) == 4

    def test_get_burnup_no_data_returns_empty(self, serpent_file):
        r = SerpentReader(serpent_file)
        burnup = r.get_burnup()
        assert len(burnup["burnup_MWd_kgU"]) == 0
        assert len(burnup["days"]) == 0

    def test_missing_detector_raises(self, serpent_file):
        r = SerpentReader(serpent_file)
        with pytest.raises(KeyError):
            r.get_tally("NONEXISTENT_DETECTOR_XYZ")

    def test_get_detector_empty_raises(self, tmp_path):
        path = str(tmp_path / "empty.det")
        with open(path, "w") as f:
            f.write("% comment only\n")
        r = SerpentReader(path)
        with pytest.raises(RuntimeError):
            r.get_detector()

    def test_summary_runs(self, serpent_file, capsys):
        SerpentReader(serpent_file).summary()
        out = capsys.readouterr().out
        assert "SERPENT2" in out

    def test_summary_shows_keff(self, serpent_file, capsys):
        SerpentReader(serpent_file).summary()
        out = capsys.readouterr().out
        assert "k-eff" in out

    def test_summary_shows_burnup(self, serpent_burnup, capsys):
        SerpentReader(serpent_burnup).summary()
        out = capsys.readouterr().out
        assert "Burnup" in out

    def test_scalar_keys(self, serpent_file):
        r = SerpentReader(serpent_file)
        skeys = r.scalar_keys()
        assert isinstance(skeys, list)
        assert "ANA_KEFF" in skeys

    def test_two_detectors_independent(self, serpent_file):
        r = SerpentReader(serpent_file)
        v_n, _ = r.get_detector("NEUTRON")
        v_g, _ = r.get_detector("GAMMA")
        assert not np.array_equal(v_n, v_g)


# ═══════════════════════════════════════════════════════════════════
# SimReader integration — all backends via unified API
# ═══════════════════════════════════════════════════════════════════


class TestSimReaderNativeBackends:
    def test_fluka_via_simreader(self, tmp_path):
        from triples_sigfast.io.sim_reader import SimReader

        path = str(tmp_path / "sim.flair")
        make_fluka_file(path)
        r = SimReader(path)
        assert r.format == "fluka"
        values, bins = r.get_spectrum()
        assert len(values) > 0

    def test_mcnp_via_simreader(self, tmp_path):
        from triples_sigfast.io.sim_reader import SimReader

        path = str(tmp_path / "sim.mctal")
        make_mctal_file(path)
        r = SimReader(path)
        assert r.format == "mcnp"
        values, bins = r.get_spectrum()
        assert len(values) > 0

    def test_serpent_via_simreader(self, tmp_path):
        from triples_sigfast.io.sim_reader import SimReader

        path = str(tmp_path / "sim.det")
        make_serpent_file(path)
        r = SimReader(path)
        assert r.format == "serpent"
        values, bins = r.get_spectrum()
        assert len(values) > 0

    def test_all_backends_have_consistent_api(self, tmp_path):
        """All backends must implement keys(), get_spectrum(), summary()."""
        from triples_sigfast.io.sim_reader import SimReader

        files = {
            "sim.flair": make_fluka_file,
            "sim.mctal": make_mctal_file,
            "sim.det": make_serpent_file,
        }
        for fname, factory in files.items():
            path = str(tmp_path / fname)
            factory(path)
            r = SimReader(path)
            assert isinstance(r.keys(), list)
            values, bins = r.get_spectrum()
            assert len(values) == len(bins)
            assert len(values) > 0
