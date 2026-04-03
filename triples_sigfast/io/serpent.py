"""
triples_sigfast.io.serpent
───────────────────────────
Native SERPENT2 detector output reader.

Parses MATLAB-compatible (.m) and detector (.det) output files produced
by SERPENT2, including:
- Detector spectra (DET_* arrays)
- k-effective evolution (ANA_KEFF, IMP_KEFF)
- Burnup data (BURNUP, BURN_DAYS)
- Group constants

Reference
---------
SERPENT2 Wiki: http://serpent.vtt.fi/mediawiki
"""

from __future__ import annotations

import re

import numpy as np

# Number of columns in a standard SERPENT2 detector array
_DET_COLS = 12


class SerpentReader:
    """
    Reader for SERPENT2 detector output files (.det, .m).

    Parses all DET_* arrays and scalar values from the MATLAB-style
    SERPENT2 output. Exposes detector spectra, k-effective, and
    burnup data through a unified API.

    Parameters
    ----------
    filepath : str
        Path to the SERPENT2 output file.

    Examples
    --------
    >>> reader = SerpentReader("serpent_det.m")
    >>> reader.summary()
    >>> flux, energies = reader.get_detector("neutron_flux")
    >>> keff = reader.get_keff()
    >>> burnup = reader.get_burnup()
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._detectors: dict[str, dict] = {}
        self._scalars: dict[str, np.ndarray] = {}
        self._parse()

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse(self) -> None:
        with open(self.filepath) as f:
            content = f.read()

        self._parse_arrays(content)
        self._classify_arrays()

    def _parse_arrays(self, content: str) -> None:
        """
        Parse all MATLAB-style array assignments: NAME = [ ... ];
        Also parses scalar assignments: NAME = value;
        """
        # Array pattern: NAME = [ numbers... ];
        array_pattern = re.compile(
            r"([A-Z][A-Z0-9_]*)\s*=\s*\[(.*?)\]\s*;",
            re.DOTALL,
        )
        for match in array_pattern.finditer(content):
            name = match.group(1)
            raw = re.findall(
                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                match.group(2),
            )
            if raw:
                self._scalars[name] = np.array(
                    [float(v) for v in raw], dtype=np.float64
                )

        # Scalar pattern: NAME = value; (not inside brackets)
        scalar_pattern = re.compile(
            r"^([A-Z][A-Z0-9_]*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;",
            re.MULTILINE,
        )
        for match in scalar_pattern.finditer(content):
            name = match.group(1)  # pragma: no cover
            if name not in self._scalars:  # pragma: no cover
                self._scalars[name] = np.array(
                    [float(match.group(2))], dtype=np.float64
                )

    def _classify_arrays(self) -> None:
        """
        Classify parsed arrays into detectors vs scalar quantities.

        SERPENT2 detector arrays have exactly N×12 elements.
        Arrays with fewer elements are treated as scalar/vector data.
        """
        for name, arr in self._scalars.items():
            n = len(arr)
            if n >= _DET_COLS and n % _DET_COLS == 0:
                mat = arr.reshape(-1, _DET_COLS)
                self._detectors[name] = {
                    "name": name,
                    "raw": mat,
                    # Standard SERPENT2 DET columns:
                    # 0=E_low, 1=E_high, 2-9=scores, 10=mean, 11=rel_err
                    "bins": 0.5 * (mat[:, 0] + mat[:, 1]),
                    "e_low": mat[:, 0],
                    "e_high": mat[:, 1],
                    "values": mat[:, 10],
                    "errors": mat[:, 11],
                }

    # ── Public API ────────────────────────────────────────────────────────

    def get_detector(self, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract (flux_values, bin_centres) from a detector.

        Parameters
        ----------
        name : str, optional
            Detector name (exact or partial, case-insensitive).
            If None, returns the first available detector.

        Returns
        -------
        values : np.ndarray
            Mean flux/reaction rate values per energy bin.
        bin_centres : np.ndarray
            Energy bin centres in MeV.

        Raises
        ------
        RuntimeError
            If no detectors are found in the file.
        KeyError
            If the named detector is not found.
        """
        if not self._detectors:
            raise RuntimeError(f"No detectors parsed from {self.filepath}")
        if name is None:
            det = next(iter(self._detectors.values()))
            return det["values"], det["bins"]
        det = self._get_detector(name)
        return det["values"], det["bins"]

    def get_spectrum(self, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Alias for get_detector — unified SimReader API."""
        return self.get_detector(name)

    def get_tally(self, name: str) -> dict:
        """
        Retrieve full detector dict by name.

        Returns
        -------
        dict with keys: 'name', 'raw', 'bins', 'e_low', 'e_high',
        'values', 'errors'.
        """
        return self._get_detector(name)

    def get_keff(self) -> dict:
        """
        Retrieve k-effective values.

        Returns
        -------
        dict with keys: 'ana_keff', 'imp_keff', 'ana_err', 'imp_err'.
        Missing values are returned as np.nan.

        Examples
        --------
        >>> keff = reader.get_keff()
        >>> print(f"k-eff = {keff['ana_keff']:.5f} ± {keff['ana_err']:.5f}")
        """

        def _get(key: str, idx: int = 0) -> float:
            arr = self._scalars.get(key)
            if arr is not None and len(arr) > idx:
                return float(arr[idx])
            return float(np.nan)

        return {
            "ana_keff": _get("ANA_KEFF"),
            "ana_err": _get("ANA_KEFF", 1),
            "imp_keff": _get("IMP_KEFF"),
            "imp_err": _get("IMP_KEFF", 1),
        }

    def get_burnup(self) -> dict:
        """
        Retrieve burnup data.

        Returns
        -------
        dict with keys: 'burnup_MWd_kgU', 'days'.
        Returns empty arrays if burnup data is not present.
        """
        burnup = self._scalars.get("BURNUP", np.array([]))
        days = self._scalars.get("BURN_DAYS", np.array([]))
        return {
            "burnup_MWd_kgU": burnup,
            "days": days,
        }

    def keys(self) -> list[str]:
        """Return all detector names."""
        return list(self._detectors.keys())

    def scalar_keys(self) -> list[str]:
        """Return all parsed scalar/vector variable names."""
        return list(self._scalars.keys())

    def summary(self) -> None:
        """Print a human-readable summary of detectors and key scalars."""
        print(f"\nSERPENT2 file: {self.filepath}")
        print(f"  Detectors: {len(self._detectors)}")
        for name, det in self._detectors.items():
            n = len(det["values"])
            max_err = det["errors"].max() if n > 0 else 0.0
            print(f"  {name:<35} {n} bins  max_err={max_err:.4f}")

        keff = self.get_keff()
        if not np.isnan(keff["ana_keff"]):
            print(
                f"\n  k-eff (analog) = {keff['ana_keff']:.6f} ± {keff['ana_err']:.6f}"
            )

        burnup = self.get_burnup()
        if len(burnup["days"]) > 0:
            print(f"  Burnup steps: {len(burnup['days'])}")
        print()

    def __repr__(self) -> str:
        return f"SerpentReader('{self.filepath}', {len(self._detectors)} detector(s))"

    def __len__(self) -> int:
        return len(self._detectors)

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_detector(self, name: str) -> dict:
        if name in self._detectors:
            return self._detectors[name]
        matches = [k for k in self._detectors if name.upper() in k.upper()]
        if not matches:
            raise KeyError(
                f"Detector '{name}' not found. "
                f"Available: {list(self._detectors.keys())}"
            )
        return self._detectors[matches[0]]
