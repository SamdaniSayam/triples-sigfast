"""
triples_sigfast.io.sim_reader
------------------------------
Universal simulation file reader.

This module provides a single entry point -- the SimReader class -- that
accepts any supported simulation output file and automatically delegates to
the appropriate backend reader.  All backends implement the same public API,
so the calling code does not need to know which simulation code produced the
file.

Supported formats
-----------------
Extension       Backend         Simulation code
-----------     -----------     ---------------
.root           RootReader      Geant4 (via uproot)
.flair          FlukaReader     FLUKA
.lis            FlukaReader     FLUKA (alternative extension)
.mctal          MCNPReader      MCNP
.det            SerpentReader   SERPENT
.m              SerpentReader   SERPENT (alternative extension)
.lhe            LHEReader       PYTHIA / MadGraph (Les Houches Event format)
.hepmc          HepMCReader     PYTHIA / Herwig (HepMC3 format)
.hepmc3         HepMCReader     HepMC3 (explicit version suffix)
.csv            RawReader       Comma-separated plain-text data
.tsv            RawReader       Tab-separated plain-text data
.txt            RawReader       Whitespace-delimited plain-text data
.dat            RawReader       Generic columnar data (lab exports, custom codes)
.asc            RawReader       ASCII spectrum exports (MCA, Ortec DAQ systems)
.out            RawReader       Generic simulation text output

Design notes
------------
- Format detection is purely extension-based (_EXT_MAP lookup); no file
  content inspection is performed.  This is fast and sufficient for all
  known simulation code conventions.
- All heavy backend imports (uproot, ROOT bindings, etc.) are deferred to
  _load_backend() and are only executed when SimReader is instantiated.
  This keeps `import triples_sigfast` instant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Format detection map
# ---------------------------------------------------------------------------
# Maps each recognised file extension to an internal format identifier.
# The identifier is used as the key for selecting the correct backend class
# in _load_backend().
# ---------------------------------------------------------------------------
_EXT_MAP: dict[str, str] = {
    # Simulation engine output formats
    ".root": "geant4",
    ".flair": "fluka",
    ".lis": "fluka",
    ".mctal": "mcnp",
    ".det": "serpent",
    ".m": "serpent",
    ".lhe": "lhe",
    ".hepmc": "hepmc",
    ".hepmc3": "hepmc",
    # Plain-text / raw columnar data formats
    ".csv": "raw",
    ".tsv": "raw",
    ".txt": "raw",
    ".dat": "raw",
    ".asc": "raw",
    ".out": "raw",
}


def _detect_format(filepath: str) -> str:
    """Return the internal format identifier for the given file path.

    Raises
    ------
    ValueError
        If the file extension is not in _EXT_MAP.
    """
    ext = Path(filepath).suffix.lower()
    if ext not in _EXT_MAP:
        raise ValueError(
            f"Unrecognised file extension '{ext}'. "
            f"Supported extensions: {sorted(_EXT_MAP.keys())}"
        )
    return _EXT_MAP[ext]


# ---------------------------------------------------------------------------
# SimReader
# ---------------------------------------------------------------------------


class SimReader:
    """Universal reader for simulation output files and raw data files.

    Automatically detects the file format from its extension and delegates
    all I/O operations to the appropriate backend reader.  The public API
    (get_spectrum, get_tally, summary, keys) is identical regardless of the
    underlying format.

    Parameters
    ----------
    filepath : str
        Path to the simulation output or data file.  The file extension
        determines which backend is loaded.

    Attributes
    ----------
    filepath : str
        The path passed to the constructor.
    format : str
        The detected format identifier (e.g., 'geant4', 'mcnp', 'raw').

    Examples
    --------
    >>> reader = SimReader("output.root")        # Geant4 via uproot
    >>> reader = SimReader("output.mctal")       # MCNP
    >>> reader = SimReader("spectrum.csv")       # plain-text CSV
    >>> reader = SimReader("tally.dat")          # generic columnar data

    >>> counts, energies = reader.get_spectrum()
    >>> tally = reader.get_tally("neutron_flux")
    >>> reader.summary()
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        # Detect format from extension; raises ValueError on unknown extension.
        self.format = _detect_format(filepath)
        # Load the appropriate backend.  All deferred imports live here.
        self._backend = self._load_backend(filepath, self.format)

    # ------------------------------------------------------------------
    # Backend factory
    # ------------------------------------------------------------------

    def _load_backend(self, filepath: str, fmt: str):
        """Instantiate and return the backend reader for the given format.

        All backend classes are imported lazily inside this method to avoid
        loading uproot, ROOT bindings, or other heavy dependencies at module
        import time.

        Parameters
        ----------
        filepath : str
            Path forwarded to the backend constructor.
        fmt : str
            Format identifier from _EXT_MAP.

        Returns
        -------
        object
            An instance of the appropriate reader class.

        Raises
        ------
        ValueError
            If `fmt` does not match any known backend (should not occur in
            normal use since _detect_format validates extensions first).
        """
        if fmt == "geant4":
            from triples_sigfast.io.root_reader import RootReader

            return RootReader(filepath)

        if fmt == "fluka":
            from triples_sigfast.io.fluka import FlukaReader

            return FlukaReader(filepath)

        if fmt == "mcnp":
            from triples_sigfast.io.mcnp import MCNPReader

            return MCNPReader(filepath)

        if fmt == "serpent":
            from triples_sigfast.io.serpent import SerpentReader

            return SerpentReader(filepath)

        if fmt == "lhe":
            from triples_sigfast.io.lhe import LHEReader

            return LHEReader(filepath)

        if fmt == "hepmc":
            from triples_sigfast.io.hepmc import HepMCReader

            return HepMCReader(filepath)

        if fmt == "raw":
            # RawReader handles all plain-text columnar formats with automatic
            # delimiter and column detection.
            from triples_sigfast.io.raw import RawReader

            return RawReader(filepath)

        # This branch is unreachable in normal use because _detect_format
        # would have already raised a ValueError for unknown extensions.
        raise ValueError(
            f"No backend registered for format identifier: '{fmt}'"
        )  # pragma: no cover

    # ------------------------------------------------------------------
    # Unified public API
    # ------------------------------------------------------------------

    def get_spectrum(
        self,
        key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a one-dimensional energy spectrum from the file.

        Parameters
        ----------
        key : str, optional
            Histogram name (ROOT), tally identifier (MCNP/SERPENT), or column
            name / 'colN' index (raw data files).  If None, the backend returns
            the first available spectrum.

        Returns
        -------
        counts : np.ndarray
            Bin counts (or fluence values) as a 1-D float64 array.
        bin_centres : np.ndarray
            Corresponding energy axis values in MeV, same length as counts.
        """
        return self._backend.get_spectrum(key)  # type: ignore[arg-type]

    def get_tally(self, name: str) -> dict:
        """Retrieve a named tally result as a dictionary.

        Returns a dict with the following keys (exact content is backend-
        dependent):
            'name'   : str           -- tally identifier
            'values' : np.ndarray    -- tally values
            'errors' : np.ndarray    -- associated relative errors (0 for raw data)
            'bins'   : np.ndarray    -- bin centres or indices

        Parameters
        ----------
        name : str
            Tally or column name.
        """
        return self._backend.get_tally(name)  # type: ignore[attr-defined]

    def summary(self) -> None:
        """Print a human-readable summary of available data in the file.

        The exact output format depends on the backend, but always includes
        the list of available histogram / tally / column names.
        """
        self._backend.summary()

    def keys(self) -> list[str]:
        """Return all available histogram, tally, or column names in the file."""
        return self._backend.keys()

    def __repr__(self) -> str:
        return f"SimReader('{self.filepath}', format='{self.format}')"
