"""
triples_sigfast.io.fluka
────────────────────────
Native FLUKA simulation output reader.

Parses USRBIN, USRBDX, and USRTRACK ASCII output produced by FLUKA
(and its GUI front-end Flair). Exposes a unified API consistent with
all other SimReader backends.

Supported FLUKA estimators
--------------------------
USRBIN   — spatial energy deposition / fluence maps
USRBDX   — boundary crossing fluence / current
USRTRACK — track-length fluence estimator

Reference
---------
FLUKA Manual: https://fluka.cern/documentation
"""

from __future__ import annotations

import numpy as np


class FlukaReader:
    """
    Reader for FLUKA ASCII output files (.flair, .lis, .out).

    Parses detector blocks produced by USRBIN, USRBDX, and USRTRACK
    estimators. Each detector is stored as a named entry accessible
    via the unified API.

    Parameters
    ----------
    filepath : str
        Path to the FLUKA output file.

    Examples
    --------
    >>> reader = FlukaReader("output.flair")
    >>> reader.summary()
    >>> fluence = reader.get_usrbin("neutron_fluence")
    >>> spectrum, energies = reader.get_spectrum("gamma_spectrum")
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._detectors: dict[str, dict] = {}
        self._parse()

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse(self) -> None:
        """Parse all detector blocks from the FLUKA output file."""
        with open(self.filepath) as f:
            content = f.read()

        self._parse_detector_blocks(content)

    def _parse_detector_blocks(self, content: str) -> None:
        """
        Parse FLUKA detector blocks.

        Supports two formats:
        1. Flair-style: # DETECTOR: name / # ESTIMATOR: type blocks
        2. Plain two-column: energy  value rows grouped by blank lines
        """
        lines = content.splitlines()

        current_name: str | None = None
        current_estimator: str = "USRTRACK"
        energies: list[float] = []
        values: list[float] = []
        errors: list[float] = []

        def _flush() -> None:
            if current_name and values:
                self._detectors[current_name] = {
                    "name": current_name,
                    "estimator": current_estimator,
                    "values": np.array(values, dtype=np.float64),
                    "errors": np.array(errors, dtype=np.float64)
                    if errors
                    else np.zeros(len(values), dtype=np.float64),
                    "bins": np.array(energies, dtype=np.float64)
                    if energies
                    else np.arange(len(values), dtype=np.float64),
                }

        for line in lines:
            stripped = line.strip()

            # Blank line flushes current detector
            if not stripped:
                _flush()
                energies, values, errors = [], [], []
                continue

            # Comment / header lines
            if stripped.startswith("#"):
                if stripped.upper().startswith("# DETECTOR:"):
                    _flush()
                    energies, values, errors = [], [], []
                    current_name = stripped.split(":", 1)[-1].strip()
                elif stripped.upper().startswith("# ESTIMATOR:"):
                    current_estimator = stripped.split(":", 1)[-1].strip().upper()
                elif stripped.upper().startswith("# USRBIN"):
                    current_estimator = "USRBIN"  # pragma: no cover
                elif stripped.upper().startswith("# USRBDX"):
                    current_estimator = "USRBDX"  # pragma: no cover
                continue

            # Data lines: 2 or 3 columns (energy, value[, error])
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    energies.append(float(parts[0]))
                    values.append(float(parts[1]))
                    if len(parts) >= 3:
                        errors.append(float(parts[2]))
                except ValueError:  # pragma: no cover
                    continue  # pragma: no cover

        _flush()

    # ── Public API ────────────────────────────────────────────────────────

    def get_usrbin(self, name: str) -> dict:
        """
        Retrieve a USRBIN spatial fluence/dose map.

        Parameters
        ----------
        name : str
            Detector name (exact or partial match).

        Returns
        -------
        dict with keys: 'name', 'estimator', 'values', 'errors', 'bins'.
        """
        return self._get_detector(name)

    def get_usrbdx(self, name: str) -> dict:
        """
        Retrieve a USRBDX boundary crossing spectrum.

        Parameters
        ----------
        name : str
            Detector name (exact or partial match).

        Returns
        -------
        dict with keys: 'name', 'estimator', 'values', 'errors', 'bins'.
        """
        return self._get_detector(name)

    def get_spectrum(self, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract (values, bin_centres) from a named detector.

        Parameters
        ----------
        name : str, optional
            Detector name. If None, returns the first available detector.

        Returns
        -------
        values : np.ndarray
        bins : np.ndarray
        """
        if not self._detectors:
            raise RuntimeError(f"No detectors parsed from {self.filepath}")
        if name is None:
            det = next(iter(self._detectors.values()))
        else:
            det = self._get_detector(name)
        return det["values"], det["bins"]

    def get_tally(self, name: str) -> dict:
        """Retrieve a detector by name (unified SimReader API)."""
        return self._get_detector(name)

    def keys(self) -> list[str]:
        """Return all detector names."""
        return list(self._detectors.keys())

    def summary(self) -> None:
        """Print a human-readable summary of all detectors."""
        print(f"\nFLUKA file: {self.filepath}")
        print(f"  Detectors parsed: {len(self._detectors)}")
        for name, det in self._detectors.items():
            n = len(det["values"])
            integral = det["values"].sum()
            print(
                f"  {name:<30} [{det['estimator']:<8}] "
                f"{n} bins, integral={integral:.4e}"
            )
        print()

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_detector(self, name: str) -> dict:
        if name in self._detectors:
            return self._detectors[name]
        matches = [k for k in self._detectors if name.lower() in k.lower()]
        if not matches:
            raise KeyError(
                f"Detector '{name}' not found. "
                f"Available: {list(self._detectors.keys())}"
            )
        return self._detectors[matches[0]]

    def __repr__(self) -> str:
        return f"FlukaReader('{self.filepath}', {len(self._detectors)} detector(s))"

    def __len__(self) -> int:
        return len(self._detectors)
