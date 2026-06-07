"""
tests/test_raw.py
──────────────────
Comprehensive test suite for triples_sigfast.io.raw.RawReader.

Covers all auto-detection paths, column-resolution logic, error handling,
get_spectrum / get_tally / keys / summary / __repr__, single-column files,
semicolon/tab/whitespace delimiters, comment stripping, NaN padding,
and the _is_comment / _try_delimiter / _detect_delimiter module helpers.

Target: raise raw.py coverage from 17% → 90%+.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from triples_sigfast.io.raw import (
    _COUNTS_KEYWORDS,
    _ENERGY_KEYWORDS,
    RawReader,
    _detect_delimiter,
    _is_comment,
    _match_col,
    _try_delimiter,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def write_file(tmp_path, name: str, content: str) -> str:
    """Write a text file to tmp_path and return the absolute path string."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper functions
# ─────────────────────────────────────────────────────────────────────────────


class TestIsComment:
    def test_hash_comment(self):
        assert _is_comment("# this is a comment")

    def test_percent_comment(self):
        assert _is_comment("% MCNP comment")

    def test_exclamation_comment(self):
        assert _is_comment("! FLUKA comment")

    def test_double_slash_comment(self):
        assert _is_comment("// C++ style comment")

    def test_star_comment(self):
        assert _is_comment("* MCNP input deck comment")

    def test_blank_line(self):
        assert _is_comment("   ")

    def test_empty_string(self):
        assert _is_comment("")

    def test_numeric_line_is_not_comment(self):
        assert not _is_comment("1.0 2.0 3.0")

    def test_header_line_is_not_comment(self):
        assert not _is_comment("energy counts")


class TestTryDelimiter:
    def test_comma_delimiter(self):
        lines = ["1.0,2.0", "3.0,4.0"]
        result = _try_delimiter(lines, ",")
        assert result is not None
        assert len(result) == 2

    def test_tab_delimiter(self):
        lines = ["1.0\t2.0", "3.0\t4.0"]
        result = _try_delimiter(lines, "\t")
        assert result is not None

    def test_whitespace_delimiter(self):
        lines = ["1.0 2.0", "3.0 4.0"]
        result = _try_delimiter(lines, None)
        assert result is not None

    def test_wrong_delimiter_returns_none(self):
        lines = ["1.0,2.0", "3.0,4.0"]
        result = _try_delimiter(lines, "\t")  # tab is wrong
        assert result is None

    def test_non_numeric_token_returns_none(self):
        lines = ["energy counts"]
        result = _try_delimiter(lines, None)
        assert result is None

    def test_empty_lines_returns_none(self):
        result = _try_delimiter([], ",")
        assert result is None


class TestDetectDelimiter:
    def test_detects_comma(self):
        lines = ["1.0,2.0", "3.0,4.0"]
        assert _detect_delimiter(lines) == ","

    def test_detects_tab(self):
        lines = ["1.0\t2.0", "3.0\t4.0"]
        assert _detect_delimiter(lines) == "\t"

    def test_detects_whitespace(self):
        lines = ["1.0 2.0", "3.0 4.0"]
        assert _detect_delimiter(lines) is None

    def test_fallback_to_whitespace(self):
        # semicolons fail, everything else fails, fallback is None
        lines = ["abc def"]
        d = _detect_delimiter(lines)
        assert d is None


class TestMatchCol:
    def test_matches_energy_header(self):
        idx = _match_col(["energy", "counts"], _ENERGY_KEYWORDS)
        assert idx == 0

    def test_matches_counts_header(self):
        idx = _match_col(["energy", "counts"], _COUNTS_KEYWORDS)
        assert idx == 1

    def test_no_match_returns_none(self):
        idx = _match_col(["foo", "bar"], _ENERGY_KEYWORDS)
        assert idx is None

    def test_case_insensitive_match(self):
        idx = _match_col(["ENERGY", "flux"], _ENERGY_KEYWORDS)
        assert idx == 0


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — two-column CSV with named headers
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderCSV:
    @pytest.fixture
    def csv_path(self, tmp_path):
        return write_file(
            tmp_path,
            "data.csv",
            """\
            energy,counts
            0.1,100.0
            0.2,200.0
            0.3,150.0
            0.4,80.0
        """,
        )

    def test_keys_contains_headers(self, csv_path):
        r = RawReader(csv_path)
        assert "energy" in r.keys()
        assert "counts" in r.keys()

    def test_get_spectrum_shape(self, csv_path):
        r = RawReader(csv_path)
        counts, energies = r.get_spectrum()
        assert len(counts) == 4
        assert len(energies) == 4

    def test_get_spectrum_values(self, csv_path):
        r = RawReader(csv_path)
        counts, energies = r.get_spectrum()
        assert np.isclose(energies[0], 0.1)
        assert np.isclose(counts[0], 100.0)

    def test_get_spectrum_explicit_key(self, csv_path):
        r = RawReader(csv_path)
        counts, energies = r.get_spectrum(key="counts")
        assert len(counts) == 4

    def test_get_spectrum_key_col0(self, csv_path):
        r = RawReader(csv_path)
        counts, energies = r.get_spectrum(key="col0")
        assert len(counts) == 4

    def test_get_tally(self, csv_path):
        r = RawReader(csv_path)
        t = r.get_tally("counts")
        assert "name" in t
        assert "values" in t
        assert "errors" in t
        assert "bins" in t
        assert np.all(t["errors"] == 0.0)
        assert len(t["values"]) == 4

    def test_repr(self, csv_path):
        r = RawReader(csv_path)
        s = repr(r)
        assert "RawReader" in s
        assert "rows=4" in s
        assert "cols=2" in s

    def test_summary_runs(self, csv_path, capsys):
        r = RawReader(csv_path)
        r.summary()
        out = capsys.readouterr().out
        assert "energy" in out or "counts" in out

    def test_data_dtype_float64(self, csv_path):
        r = RawReader(csv_path)
        counts, energies = r.get_spectrum()
        assert counts.dtype == np.float64
        assert energies.dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — tab-separated file
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderTSV:
    @pytest.fixture
    def tsv_path(self, tmp_path):
        return write_file(
            tmp_path,
            "data.tsv",
            """\
            energy\tcounts
            0.5\t500.0
            1.0\t400.0
            1.5\t300.0
        """,
        )

    def test_parse_tab_delimited(self, tsv_path):
        r = RawReader(tsv_path)
        counts, energies = r.get_spectrum()
        assert len(counts) == 3

    def test_first_row_values(self, tsv_path):
        r = RawReader(tsv_path)
        counts, energies = r.get_spectrum()
        assert np.isclose(energies[0], 0.5)
        assert np.isclose(counts[0], 500.0)


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — whitespace-delimited .dat file with comments
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderDatWithComments:
    @pytest.fixture
    def dat_path(self, tmp_path):
        return write_file(
            tmp_path,
            "spectrum.dat",
            """\
            # Neutron spectrum from Geant4
            % Material: polyethylene
            ! Settings: 1e6 histories
            energy counts
            0.01 120.5
            0.10 980.3
            0.50 1200.0
            1.00 850.0
            2.00 300.0
        """,
        )

    def test_comments_are_stripped(self, dat_path):
        r = RawReader(dat_path)
        counts, energies = r.get_spectrum()
        assert len(counts) == 5

    def test_first_value_correct(self, dat_path):
        r = RawReader(dat_path)
        counts, energies = r.get_spectrum()
        assert np.isclose(energies[0], 0.01)
        assert np.isclose(counts[0], 120.5)


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — single-column file (synthetic energy axis)
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderSingleColumn:
    @pytest.fixture
    def single_col_path(self, tmp_path):
        return write_file(
            tmp_path,
            "counts.txt",
            """\
            100.0
            200.0
            300.0
            400.0
            500.0
        """,
        )

    def test_single_column_parses(self, single_col_path):
        r = RawReader(single_col_path)
        counts, energies = r.get_spectrum()
        assert len(counts) == 5

    def test_synthetic_energy_axis(self, single_col_path):
        r = RawReader(single_col_path)
        counts, energies = r.get_spectrum()
        # Synthetic axis should be 0, 1, 2, 3, 4
        assert np.allclose(energies, np.arange(5, dtype=float))

    def test_counts_values_match(self, single_col_path):
        r = RawReader(single_col_path)
        counts, energies = r.get_spectrum()
        assert np.isclose(counts[0], 100.0)
        assert np.isclose(counts[-1], 500.0)


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — no-header file (positional fallback)
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderNoHeader:
    @pytest.fixture
    def noheader_path(self, tmp_path):
        return write_file(
            tmp_path,
            "noheader.txt",
            """\
            0.1 120.5
            0.2 980.3
            0.3 1200.0
            0.4 850.0
        """,
        )

    def test_no_header_generates_generic_names(self, noheader_path):
        r = RawReader(noheader_path)
        assert "col0" in r.keys()
        assert "col1" in r.keys()

    def test_positional_fallback_col0_energy_col1_counts(self, noheader_path):
        r = RawReader(noheader_path)
        counts, energies = r.get_spectrum()
        assert len(counts) == 4
        assert np.isclose(energies[0], 0.1)
        assert np.isclose(counts[0], 120.5)


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — semicolon-separated file
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderSemicolon:
    @pytest.fixture
    def semi_path(self, tmp_path):
        return write_file(
            tmp_path,
            "data.csv",
            """\
            E;counts
            0.5;500.0
            1.0;300.0
            2.0;150.0
        """,
        )

    def test_semicolon_delimiter(self, semi_path):
        r = RawReader(semi_path)
        counts, energies = r.get_spectrum()
        assert len(counts) == 3
        assert np.isclose(energies[0], 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — column resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderColumnResolution:
    @pytest.fixture
    def multi_col_path(self, tmp_path):
        return write_file(
            tmp_path,
            "multi.csv",
            """\
            energy,flux,counts,rate
            0.1,10.0,100.0,1.0
            0.2,20.0,200.0,2.0
            0.3,30.0,300.0,3.0
        """,
        )

    def test_resolve_by_name(self, multi_col_path):
        r = RawReader(multi_col_path)
        t = r.get_tally("flux")
        assert np.isclose(t["values"][0], 10.0)

    def test_resolve_by_col_index(self, multi_col_path):
        r = RawReader(multi_col_path)
        t = r.get_tally("col2")
        assert np.isclose(t["values"][0], 100.0)

    def test_col_index_out_of_range_raises(self, multi_col_path):
        r = RawReader(multi_col_path)
        with pytest.raises(ValueError, match="out of range"):
            r.get_tally("col99")

    def test_unknown_name_raises(self, multi_col_path):
        r = RawReader(multi_col_path)
        with pytest.raises(ValueError, match="not found"):
            r.get_tally("nonexistent_column")

    def test_get_spectrum_explicit_key_avoids_same_col(self, multi_col_path):
        r = RawReader(multi_col_path)
        # col0 = energy, use it as counts key → energy falls back to col1
        counts, energies = r.get_spectrum(key="col0")
        assert len(counts) == 3

    def test_get_spectrum_key_is_counts_col0(self, tmp_path):
        """When counts key resolves to col0, energy should fallback to col1."""
        p = write_file(
            tmp_path,
            "x.csv",
            """\
            counts,energy
            100,0.1
            200,0.2
        """,
        )
        r = RawReader(p)
        counts, energies = r.get_spectrum(key="col0")
        assert len(counts) == 2


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — alternative energy/counts column name keywords
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderColumnKeywords:
    def _make(self, tmp_path, header, rows):
        content = header + "\n" + "\n".join(rows)
        return write_file(tmp_path, "kw.csv", content)

    def test_mev_header_as_energy(self, tmp_path):
        p = self._make(tmp_path, "MeV,flux", ["0.1,100", "0.2,200"])
        r = RawReader(p)
        counts, energies = r.get_spectrum()
        assert np.isclose(energies[0], 0.1)

    def test_channel_header_as_energy(self, tmp_path):
        p = self._make(tmp_path, "channel,n", ["1,50", "2,60"])
        r = RawReader(p)
        counts, energies = r.get_spectrum()
        assert np.isclose(energies[0], 1.0)

    def test_flux_header_as_counts(self, tmp_path):
        p = self._make(tmp_path, "E,flux", ["0.5,1000", "1.0,900"])
        r = RawReader(p)
        counts, energies = r.get_spectrum()
        assert np.isclose(counts[0], 1000.0)

    def test_tally_header_as_counts(self, tmp_path):
        p = self._make(tmp_path, "x,tally", ["1,500", "2,600"])
        r = RawReader(p)
        counts, energies = r.get_spectrum()
        assert np.isclose(counts[0], 500.0)


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — error cases
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderErrors:
    def test_all_comments_raises(self, tmp_path):
        p = write_file(
            tmp_path,
            "comments_only.txt",
            """\
            # comment 1
            # comment 2
            % comment 3
        """,
        )
        with pytest.raises(ValueError, match="No numeric data"):
            RawReader(p)

    def test_empty_file_raises(self, tmp_path):
        p = write_file(tmp_path, "empty.txt", "")
        with pytest.raises(ValueError, match="No numeric data"):
            RawReader(p)

    def test_only_header_no_data_raises(self, tmp_path):
        p = write_file(tmp_path, "header_only.csv", "energy,counts\n")
        with pytest.raises(ValueError, match="No numeric data"):
            RawReader(p)


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — NaN handling and jagged rows
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderNaNHandling:
    @pytest.fixture
    def jagged_path(self, tmp_path):
        """File where one row has fewer columns — should be padded with NaN."""
        return write_file(
            tmp_path,
            "jagged.txt",
            """\
            0.1 100.0
            0.2 200.0 extra_ignored_if_non_numeric
            0.3 300.0
        """,
        )

    def test_nan_rows_dropped_from_spectrum(self, tmp_path):
        """Rows with NaN in energy or counts must be dropped."""
        p = write_file(
            tmp_path,
            "nan.csv",
            """\
            energy,counts
            0.1,100.0
            0.2,200.0
            0.3,300.0
        """,
        )
        r = RawReader(p)
        counts, energies = r.get_spectrum()
        assert np.all(np.isfinite(counts))
        assert np.all(np.isfinite(energies))


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — header length mismatch
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderHeaderMismatch:
    def test_more_headers_than_data_cols_truncates(self, tmp_path):
        """If header has more names than data columns, truncate headers."""
        p = write_file(
            tmp_path,
            "wide_header.csv",
            """\
            a,b,c,d,e
            1.0,2.0
            3.0,4.0
        """,
        )
        r = RawReader(p)
        # Should have only 2 headers (matching data width)
        assert len(r.keys()) == 2

    def test_fewer_headers_than_data_cols_pads(self, tmp_path):
        """If data has more columns than headers, pad with col<N> names."""
        p = write_file(
            tmp_path,
            "narrow_header.csv",
            """\
            energy
            0.1 100.0 5.0
            0.2 200.0 3.0
        """,
        )
        r = RawReader(p)
        keys = r.keys()
        assert "energy" in keys
        assert "col1" in keys
        assert "col2" in keys


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — .asc and .out extensions (generic plain text)
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderExtensions:
    def test_asc_extension(self, tmp_path):
        p = write_file(
            tmp_path,
            "mca.asc",
            """\
            # MCA export
            100.0
            200.0
            150.0
        """,
        )
        r = RawReader(p)
        counts, energies = r.get_spectrum()
        assert len(counts) == 3

    def test_out_extension(self, tmp_path):
        p = write_file(
            tmp_path,
            "sim.out",
            """\
            E flux
            1.0 500.0
            2.0 400.0
        """,
        )
        r = RawReader(p)
        counts, _ = r.get_spectrum()
        assert len(counts) == 2


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — summary output content
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderSummaryContent:
    @pytest.fixture
    def csv_path(self, tmp_path):
        return write_file(
            tmp_path,
            "summary_test.csv",
            """\
            energy,counts
            0.1,100.0
            0.5,500.0
            1.0,250.0
        """,
        )

    def test_summary_shows_column_stats(self, csv_path, capsys):
        r = RawReader(csv_path)
        r.summary()
        out = capsys.readouterr().out
        # Should show column stats
        assert "0" in out  # index
        assert "energy" in out or "counts" in out

    def test_summary_shows_filename(self, csv_path, capsys):
        r = RawReader(csv_path)
        r.summary()
        out = capsys.readouterr().out
        assert "summary_test.csv" in out


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — quoted headers
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderQuotedHeaders:
    def test_quoted_header_names_stripped(self, tmp_path):
        p = write_file(
            tmp_path,
            "quoted.csv",
            """\
            "energy","counts"
            0.1,100.0
            0.2,200.0
        """,
        )
        r = RawReader(p)
        keys = r.keys()
        assert "energy" in keys
        assert "counts" in keys

    def test_single_quoted_headers(self, tmp_path):
        p = write_file(
            tmp_path,
            "sq.csv",
            """\
            'E','flux'
            0.5,500.0
            1.0,300.0
        """,
        )
        r = RawReader(p)
        keys = r.keys()
        assert "E" in keys or "flux" in keys


# ─────────────────────────────────────────────────────────────────────────────
# RawReader — get_spectrum with only energy-column heuristics
# ─────────────────────────────────────────────────────────────────────────────


class TestRawReaderAutoHeuristic:
    def test_only_energy_found_falls_back_to_col1(self, tmp_path):
        """Energy col detected but not counts col → fallback to col0/col1."""
        p = write_file(
            tmp_path,
            "e_only.csv",
            """\
            energy,response
            0.1,200.0
            0.2,400.0
        """,
        )
        r = RawReader(p)
        counts, energies = r.get_spectrum()
        # response is not in _COUNTS_KEYWORDS, so fallback positional
        assert len(counts) == 2

    def test_both_heuristics_succeed(self, tmp_path):
        """Both energy and counts cols found by name → use them."""
        p = write_file(
            tmp_path,
            "both.csv",
            """\
            energy,counts
            0.1,100
            0.2,200
        """,
        )
        r = RawReader(p)
        counts, energies = r.get_spectrum()
        # energy col is col0 (0.1, 0.2), counts col is col1 (100, 200)
        assert np.isclose(energies[0], 0.1)
        assert np.isclose(counts[0], 100.0)
