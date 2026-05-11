"""
triples_sigfast.io.raw
-----------------------
Universal reader for plain-text columnar data files.

This module allows users to run `sigfast analyze` on any plain-text data file
without writing a single line of Python.  It auto-detects the file's delimiter,
header line, and which columns contain the energy axis and counts data.

Supported file extensions (registered in sim_reader._EXT_MAP)
-------------------------------------------------------------
.csv   -- comma-separated values
.tsv   -- tab-separated values
.txt   -- whitespace-delimited text
.dat   -- generic columnar data (nuclear lab exports, custom simulation codes)
.asc   -- ASCII spectrum exports (MCA systems, Ortec DAQ)
.out   -- generic simulation text output

Auto-detection algorithm
------------------------
1. Comment filtering : Lines beginning with '#', '%', '!', '//', or '*' are
   stripped.  Blank lines are also discarded.

2. Header detection  : The first non-comment line that contains any alphabetic
   character is treated as a header row.  All subsequent lines are treated as
   numeric data.

3. Delimiter         : Candidates are tried in order: ',' (CSV), TAB (TSV),
   whitespace (generic), ';' (semicolon-separated).  The first delimiter that
   successfully parses all data lines as fully numeric is selected.

4. Energy column     : Column header is matched against a list of common energy-
   related names (energy, E, bin, bin_centre, x, MeV, keV, eV, channel).

5. Counts column     : Column header is matched against count-related names
   (counts, n, flux, fluence, tally, y, yield, signal, intensity, value, rate,
   response).

6. Fallback          : If no headers match, col[0] is used as the energy axis
   and col[1] as counts.  For single-column files, a synthetic integer axis
   is generated.

7. Key override      : The --key CLI option accepts 'colN' (e.g., 'col2') for
   index-based column selection, or an exact header name for named selection.

Usage
-----
>>> reader = RawReader("spectrum.csv")
>>> counts, energies = reader.get_spectrum()

>>> # Select a specific column for counts
>>> counts, energies = reader.get_spectrum(key="flux")
>>> counts, energies = reader.get_spectrum(key="col2")

>>> reader.summary()    # prints a rich table of all columns
>>> reader.keys()       # returns a list of column names
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

# Module-level console used by summary().
_console = Console()


# ---------------------------------------------------------------------------
# Column-name recognition patterns
# ---------------------------------------------------------------------------
# These compiled regular expressions are matched against column header names
# to automatically identify which column is the energy axis and which is
# the counts/flux column.

_ENERGY_KEYWORDS = re.compile(
    r"^(e|en|energy|energies|bin|bin_centre|bin_center|x|mev|kev|ev|channel)$",
    re.IGNORECASE,
)
_COUNTS_KEYWORDS = re.compile(
    r"^(n|count|counts|flux|fluence|tally|y|yield|signal|intensity|value|val|rate|response)$",
    re.IGNORECASE,
)

# Characters that mark an entire line as a comment.
_COMMENT_CHARS = ("#", "%", "!", "//", "*")

# Delimiter candidates tried in priority order.
# None signals that Python's str.split() (whitespace splitting) should be used.
_DELIMITERS = [",", "\t", None, ";"]


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def _is_comment(line: str) -> bool:
    """Return True if the stripped line is a comment or blank.

    A line is considered a comment if it begins with any character in
    _COMMENT_CHARS, or if it is entirely whitespace.
    """
    stripped = line.strip()
    return any(stripped.startswith(c) for c in _COMMENT_CHARS) or stripped == ""


def _try_delimiter(lines: list[str], delim) -> list[list[str]] | None:
    """Attempt to split every line in `lines` using `delim`.

    Returns the list of split rows if every token on every line can be
    parsed as a float.  Returns None if any line contains a non-numeric
    token, which indicates the wrong delimiter was used.

    Parameters
    ----------
    lines : list[str]
        The data lines to test (comment lines already removed).
    delim : str or None
        The delimiter character to test.  None means whitespace splitting.
    """
    rows = []
    for line in lines:
        parts = line.strip().split(delim) if delim else line.strip().split()
        try:
            # All parts must convert to float; non-numeric tokens raise ValueError.
            [float(p) for p in parts if p.strip()]
            rows.append(parts)
        except ValueError:
            return None  # Wrong delimiter: at least one token was non-numeric.
    return rows if rows else None


def _detect_delimiter(data_lines: list[str]):
    """Select the best delimiter for the given data lines.

    Tries each delimiter in _DELIMITERS in order and returns the first one
    that successfully parses all data lines as numeric.

    Parameters
    ----------
    data_lines : list[str]
        Non-comment, non-blank lines from the file.

    Returns
    -------
    str or None
        The delimiter character, or None to use whitespace splitting.
    """
    for delim in _DELIMITERS:
        if _try_delimiter(data_lines, delim) is not None:
            return delim
    # If no candidate succeeded, fall back to whitespace splitting.
    return None


def _match_col(headers: list[str], pattern: re.Pattern) -> int | None:
    """Return the index of the first header that matches `pattern`.

    Parameters
    ----------
    headers : list[str]
        The list of column header names from the file.
    pattern : re.Pattern
        A compiled regular expression to match against each header.

    Returns
    -------
    int or None
        Column index of the first match, or None if no header matches.
    """
    for i, header in enumerate(headers):
        if pattern.match(header.strip()):
            return i
    return None


# ---------------------------------------------------------------------------
# RawReader class
# ---------------------------------------------------------------------------

class RawReader:
    """Universal reader for plain-text columnar data files.

    Implements the same public API as all other triples-sigfast reader backends
    (get_spectrum, get_tally, summary, keys), allowing it to be used
    transparently via SimReader.

    Parameters
    ----------
    filepath : str
        Path to the plain-text data file.

    Attributes
    ----------
    filepath : str
        The path passed to the constructor.

    Examples
    --------
    >>> r = RawReader("spectrum.csv")
    >>> counts, energies = r.get_spectrum()
    >>> r.summary()
    """

    def __init__(self, filepath: str) -> None:
        self.filepath  = filepath
        self._path     = Path(filepath)

        # Read the entire file as lines; 'replace' handles non-UTF-8 bytes
        # commonly found in legacy DAQ exports.
        self._raw_lines: list[str] = self._path.read_text(errors="replace").splitlines()

        # These are populated by _parse().
        self._headers: list[str]  = []
        self._data: np.ndarray    = np.empty((0, 0))  # shape: (n_rows, n_cols)
        self._delimiter           = None

        # Parse the file immediately on construction so that any format errors
        # are raised before the caller attempts to use get_spectrum().
        self._parse()

    # ------------------------------------------------------------------
    # Internal parsing pipeline
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        """Parse the file into a numeric data matrix.

        This method implements the full auto-detection pipeline described in
        the module docstring:
        1. Strip comments and blank lines.
        2. Identify the optional header row.
        3. Detect the column delimiter.
        4. Build the numeric data matrix (n_rows x n_cols).
        5. Reconcile the header list length with the column count.
        """
        potential_header: str | None = None
        data_lines: list[str]        = []

        for line in self._raw_lines:
            stripped = line.strip()
            if not stripped:
                continue  # Skip blank lines.
            if _is_comment(stripped):
                continue  # Skip comment lines.

            # A line is a header if it contains at least one alphabetic token
            # and no data lines have been collected yet.  This correctly handles
            # files where the header is the very first non-comment line.
            parts     = stripped.split(",") if "," in stripped else stripped.split()
            has_alpha = any(re.search(r"[a-zA-Z]", p) for p in parts)

            if has_alpha and not data_lines:
                potential_header = stripped
            else:
                data_lines.append(stripped)

        if not data_lines:
            raise ValueError(
                f"No numeric data found in '{self._path.name}'. "
                "Verify that the file contains columnar numeric data."
            )

        # Detect the delimiter from the data lines.
        self._delimiter = _detect_delimiter(data_lines)

        # Parse the header row using the detected delimiter.
        if potential_header is not None:
            delim = self._delimiter
            raw_headers = (
                potential_header.strip().split(delim)
                if delim
                else potential_header.strip().split()
            )
            # Strip surrounding whitespace and quotation marks from each name.
            self._headers = [h.strip().strip('"').strip("'") for h in raw_headers]

        # Build the numeric data matrix row by row.
        rows: list[list[float]] = []
        delim = self._delimiter
        for line in data_lines:
            parts = line.strip().split(delim) if delim else line.strip().split()
            try:
                rows.append([float(p) for p in parts if p.strip()])
            except ValueError:
                continue  # Skip lines that cannot be fully parsed as numeric.

        if not rows:
            raise ValueError(
                f"Could not parse any numeric rows from '{self._path.name}'."
            )

        # Pad all rows to the same width (the maximum observed column count).
        # Shorter rows are padded with NaN so the matrix is rectangular.
        max_cols = max(len(r) for r in rows)
        padded   = [r + [np.nan] * (max_cols - len(r)) for r in rows]
        self._data = np.array(padded, dtype=float)

        # Reconcile the header list length with the actual column count.
        n_cols = self._data.shape[1]
        if not self._headers:
            # No header detected: generate generic names 'col0', 'col1', ...
            self._headers = [f"col{i}" for i in range(n_cols)]
        elif len(self._headers) < n_cols:
            # Header row has fewer names than data columns: pad with generics.
            for i in range(len(self._headers), n_cols):
                self._headers.append(f"col{i}")
        elif len(self._headers) > n_cols:
            # Header row has more names than data columns: truncate.
            self._headers = self._headers[:n_cols]

    # ------------------------------------------------------------------
    # Column resolution helper
    # ------------------------------------------------------------------

    def _resolve_col(self, key: str | None) -> int | None:
        """Convert a user-supplied key string to a zero-based column index.

        Accepted key formats:
        - 'colN'         : selects column by index (e.g., 'col0', 'col2')
        - '<header_name>': selects column by exact case-insensitive header match

        Parameters
        ----------
        key : str or None
            The key string supplied by the caller.  If None, returns None.

        Returns
        -------
        int or None
            The resolved zero-based column index, or None if key is None.

        Raises
        ------
        ValueError
            If the index is out of range or the name is not found.
        """
        if key is None:
            return None

        # Check for the 'colN' index shorthand.
        m = re.fullmatch(r"col(\d+)", key, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if idx >= self._data.shape[1]:
                raise ValueError(
                    f"Column index {idx} out of range "
                    f"(file has {self._data.shape[1]} columns: {self._headers})."
                )
            return idx

        # Fall back to case-insensitive header name search.
        for i, header in enumerate(self._headers):
            if header.lower() == key.lower():
                return i

        raise ValueError(
            f"Key '{key}' not found in this file. "
            f"Available columns: {self._headers}. "
            "Use 'col0', 'col1', etc. for index-based selection."
        )

    def _auto_energy_col(self) -> int | None:
        """Return the index of the column whose name matches _ENERGY_KEYWORDS."""
        return _match_col(self._headers, _ENERGY_KEYWORDS)

    def _auto_counts_col(self) -> int | None:
        """Return the index of the column whose name matches _COUNTS_KEYWORDS."""
        return _match_col(self._headers, _COUNTS_KEYWORDS)

    # ------------------------------------------------------------------
    # Public API (SimReader contract)
    # ------------------------------------------------------------------

    def get_spectrum(
        self,
        key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (counts, bin_centres) from the file.

        Column selection follows the priority order described in the module
        docstring: explicit key > header heuristics > column-index fallback.

        Parameters
        ----------
        key : str, optional
            Column name or 'colN' index that selects the **counts** column.
            The energy column is chosen independently by heuristic or default.

        Returns
        -------
        counts : np.ndarray
            The counts (or flux/tally) column values as float64.
        bin_centres : np.ndarray
            The corresponding energy axis values.  Units are those of the
            original file (typically MeV for nuclear physics data).

        Notes
        -----
        Rows containing NaN in either the counts or energy column are silently
        dropped before the arrays are returned.
        """
        n_cols = self._data.shape[1]

        if n_cols == 1:
            # Single-column file: interpret the one column as counts and
            # generate a synthetic integer energy axis (bin indices).
            counts   = self._data[:, 0]
            energies = np.arange(len(self._data), dtype=float)

        elif key is not None:
            # Caller specified a counts column explicitly via --key.
            counts_col = self._resolve_col(key)
            counts     = self._data[:, counts_col]

            # Try to find an energy column via heuristics; avoid picking the
            # same column as the counts column.
            energy_col = self._auto_energy_col()
            if energy_col is None or energy_col == counts_col:
                # Default: use col[0] as energy (or col[1] if counts is col[0]).
                energy_col = 0 if counts_col != 0 else 1
            energies = self._data[:, energy_col]

        else:
            # No key specified: use header heuristics to find both columns.
            energy_col = self._auto_energy_col()
            counts_col = self._auto_counts_col()

            if energy_col is not None and counts_col is not None:
                # Both columns identified by name.
                energies = self._data[:, energy_col]
                counts   = self._data[:, counts_col]
            elif n_cols >= 2:
                # Heuristic failed: fall back to positional convention
                # (column 0 = energy, column 1 = counts).
                energies = self._data[:, 0]
                counts   = self._data[:, 1]
            else:
                # Single effective column after detection.
                energies = np.arange(len(self._data), dtype=float)
                counts   = self._data[:, 0]

        # Remove any rows where either array contains NaN or inf.
        mask = np.isfinite(counts) & np.isfinite(energies)
        return counts[mask].astype(float), energies[mask].astype(float)

    def get_tally(self, name: str) -> dict:
        """Return a named column as a tally dictionary.

        Provides compatibility with the SimReader tally API.  Because raw
        data files do not carry statistical error estimates, the 'errors'
        field is always a zero array of the same shape as 'values'.

        Parameters
        ----------
        name : str
            Column name or 'colN' index.

        Returns
        -------
        dict with keys:
            'name'   -- the original name argument
            'values' -- column values as float64 (NaN rows dropped)
            'errors' -- zero array (no error estimate available)
            'bins'   -- synthetic integer indices
        """
        col    = self._resolve_col(name)
        values = self._data[:, col]
        mask   = np.isfinite(values)
        vals   = values[mask]
        return {
            "name":   name,
            "values": vals,
            "errors": np.zeros_like(vals),  # Raw data carries no statistical errors.
            "bins":   np.arange(len(vals), dtype=float),
        }

    def keys(self) -> list[str]:
        """Return the list of all column names detected in the file."""
        return list(self._headers)

    def summary(self) -> None:
        """Print a formatted table summarising all columns in the file.

        Displays the column index, name, minimum, maximum, mean, and the
        count of non-NaN rows for each column.
        """
        table = Table(
            title=f"Raw Data Summary: {self._path.name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Index",        style="dim", justify="right")
        table.add_column("Column",       style="cyan")
        table.add_column("Min",          justify="right")
        table.add_column("Max",          justify="right")
        table.add_column("Mean",         justify="right")
        table.add_column("Non-NaN Rows", justify="right")

        for i, header in enumerate(self._headers):
            col   = self._data[:, i]
            valid = col[np.isfinite(col)]
            table.add_row(
                str(i),
                header,
                f"{valid.min():.4g}"  if len(valid) else "N/A",
                f"{valid.max():.4g}"  if len(valid) else "N/A",
                f"{valid.mean():.4g}" if len(valid) else "N/A",
                str(len(valid)),
            )

        _console.print(
            f"\n[dim]File:[/dim] {self.filepath}  "
            f"[dim]Rows:[/dim] {self._data.shape[0]}  "
            f"[dim]Cols:[/dim] {self._data.shape[1]}  "
            f"[dim]Delimiter:[/dim] "
            f"{'<whitespace>' if self._delimiter is None else repr(self._delimiter)}"
        )
        _console.print(table)

    def __repr__(self) -> str:
        return (
            f"RawReader('{self.filepath}', "
            f"rows={self._data.shape[0]}, "
            f"cols={self._data.shape[1]}, "
            f"headers={self._headers})"
        )
