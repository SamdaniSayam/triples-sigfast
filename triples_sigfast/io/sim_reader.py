"""
triples_sigfast.io.sim_reader
─────────────────────────────
Universal simulation file reader.

Detects the output format of Geant4, FLUKA, MCNP, and SERPENT automatically
from the file extension and exposes a single consistent API regardless of
the underlying format.

Supported formats
-----------------
.root           -> Geant4 (via RootReader / uproot)
.flair / .lis   -> FLUKA  (via FlukaReader)
.mctal          -> MCNP   (via MCNPReader)
.det / .m       -> SERPENT (via SerpentReader)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Format detection ──────────────────────────────────────────────────────────

_EXT_MAP: dict[str, str] = {
    ".root": "geant4",
    ".flair": "fluka",
    ".lis": "fluka",
    ".mctal": "mcnp",
    ".det": "serpent",
    ".m": "serpent",
}


def _detect_format(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext not in _EXT_MAP:
        raise ValueError(
            f"Unrecognised file extension '{ext}'. Supported: {list(_EXT_MAP.keys())}"
        )
    return _EXT_MAP[ext]


# ── SimReader ─────────────────────────────────────────────────────────────────


class SimReader:
    """
    Universal reader for simulation output files.

    Automatically detects the simulation code from the file extension and
    delegates to the appropriate backend. All backends expose the same API.

    Parameters
    ----------
    filepath : str
        Path to the simulation output file. Extension determines the backend:
        .root -> Geant4, .flair/.lis -> FLUKA, .mctal -> MCNP, .det/.m -> SERPENT.

    Examples
    --------
    >>> reader = SimReader("output.root")    # Geant4
    >>> reader = SimReader("output.flair")   # FLUKA
    >>> reader = SimReader("output.mctal")   # MCNP
    >>> reader = SimReader("output.det")     # SERPENT

    >>> spectrum = reader.get_spectrum()
    >>> tally    = reader.get_tally("neutron_flux")
    >>> reader.summary()
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.format = _detect_format(filepath)
        self._backend = self._load_backend(filepath, self.format)

    def _load_backend(self, filepath: str, fmt: str):
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
        raise ValueError(f"No backend for format: {fmt}")  # pragma: no cover

    # ── Unified API ───────────────────────────────────────────────────────

    def get_spectrum(
        self,
        key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1-D energy spectrum as (counts, bin_centres).

        Parameters
        ----------
        key : str, optional
            Histogram / tally name. If None, returns the first available
            spectrum in the file.

        Returns
        -------
        counts : np.ndarray
        bin_centres : np.ndarray
        """
        return self._backend.get_spectrum(key)  # type: ignore[arg-type]

    def get_tally(self, name: str) -> dict:
        """
        Retrieve a named tally result.

        Returns a dict with keys: 'values', 'errors', 'bins', 'name'.
        The exact content depends on the simulation code.
        """
        return self._backend.get_tally(name)  # type: ignore[attr-defined]

    def summary(self) -> None:
        """Print a human-readable summary of available data in the file."""
        self._backend.summary()

    def keys(self) -> list[str]:
        """Return all available keys / tally names in the file."""
        return self._backend.keys()

    def __repr__(self) -> str:
        return f"SimReader('{self.filepath}', format='{self.format}')"
