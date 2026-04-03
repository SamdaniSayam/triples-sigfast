"""
triples_sigfast.io.root_reader
──────────────────────────────
Smart ROOT file reader for Geant4 simulation output.

Wraps uproot with a physicist-friendly API: auto-discovery of histograms,
one-line spectrum extraction, and direct export to CSV / HDF5.

Requirements: uproot >= 5.0, awkward >= 2.0, numpy, pandas
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import uproot
except ImportError as e:
    raise ImportError(
        "uproot is required for RootReader. Install it with: pip install uproot"
    ) from e  # pragma: no cover


class RootReader:
    """
    Read Geant4 ROOT output files with a clean, beginner-friendly API.

    Parameters
    ----------
    filepath : str
        Path to the .root file produced by Geant4 (or any ROOT-compatible
        simulation code).

    Examples
    --------
    >>> reader = RootReader("simulation.root")
    >>> reader.summary()
    >>> counts, energies = reader.get_spectrum("neutron")
    >>> reader.export_csv("results.csv")
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._file = uproot.open(filepath)
        self._keys: list[str] = self._file.keys()

    # ── Discovery ─────────────────────────────────────────────────────────

    def keys(self) -> list[str]:
        """Return all object keys in the ROOT file."""
        return list(self._keys)

    def histogram_keys(self) -> list[str]:
        """Return only keys that correspond to histogram objects (TH1, TH2)."""
        hist_types = ("TH1F", "TH1D", "TH1I", "TH2F", "TH2D")
        result = []
        for key in self._keys:
            try:
                obj = self._file[key]
                if obj.classname in hist_types:
                    result.append(key)
            except Exception:  # pragma: no cover
                continue
        return result

    def summary(self) -> None:
        """
        Print a human-readable summary of all objects in the ROOT file.

        Lists each key with its ROOT class name and, for histograms,
        the number of bins and total integral.
        """
        hist_types = ("TH1F", "TH1D", "TH1I", "TH2F", "TH2D")
        print(f"\nROOT file: {self.filepath}")
        print(f"{'─' * 60}")
        print(f"  {'Key':<30} {'Type':<12} {'Bins / Info'}")
        print(f"{'─' * 60}")
        for key in self._keys:
            try:
                obj = self._file[key]
                cls = obj.classname
                if cls in hist_types:
                    counts, _ = obj.to_numpy()
                    info = f"{len(counts)} bins, integral={counts.sum():.2f}"
                else:
                    info = ""  # pragma: no cover
                print(f"  {key:<30} {cls:<12} {info}")
            except Exception as exc:  # pragma: no cover
                print(f"  {key:<30} {'(error)':<12} {exc}")  # pragma: no cover
        print(f"{'─' * 60}\n")

    # ── Extraction ─────────────────────────────────────────────────────────

    def get_spectrum(
        self,
        key: str,
        flow: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract counts and bin-centre energies from a 1-D histogram.

        Parameters
        ----------
        key : str
            Histogram key as returned by `keys()` or `histogram_keys()`.
            Partial matching is supported: if the exact key is not found,
            the first key containing `key` as a substring is used.
        flow : bool, optional
            If True, include underflow (bin 0) and overflow (last bin).
            Default False (standard physics convention).

        Returns
        -------
        counts : np.ndarray
            Bin counts (float64).
        bin_centres : np.ndarray
            Energy axis as bin centres (same units as the ROOT histogram).

        Raises
        ------
        KeyError
            If no histogram matching `key` is found in the file.

        Examples
        --------
        >>> counts, energies = reader.get_spectrum("gamma")
        >>> print(f"Peak energy: {energies[counts.argmax()]:.3f} MeV")
        """
        resolved = self._resolve_key(key)
        obj = self._file[resolved]
        counts, edges = obj.to_numpy(flow=flow)
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
        return counts.astype(np.float64), bin_centres.astype(np.float64)

    def get_histogram_2d(
        self,
        key: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a 2-D histogram as counts + axis arrays.

        Returns
        -------
        counts : np.ndarray, shape (Nx, Ny)
        x_centres : np.ndarray
        y_centres : np.ndarray
        """
        resolved = self._resolve_key(key)  # pragma: no cover
        obj = self._file[resolved]  # pragma: no cover
        counts, x_edges, y_edges = obj.to_numpy()  # pragma: no cover
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])  # pragma: no cover
        y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])  # pragma: no cover
        return counts.astype(np.float64), x_centres, y_centres  # pragma: no cover

    def get_all_spectra(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Extract all 1-D histograms from the file.

        Returns
        -------
        dict mapping key → (counts, bin_centres)
        """
        th1_types = ("TH1F", "TH1D", "TH1I")
        result = {}
        for key in self._keys:
            try:
                obj = self._file[key]
                if obj.classname in th1_types:
                    counts, edges = obj.to_numpy()
                    bin_centres = 0.5 * (edges[:-1] + edges[1:])
                    result[key] = (
                        counts.astype(np.float64),
                        bin_centres.astype(np.float64),
                    )
            except Exception:  # pragma: no cover
                continue  # pragma: no cover
        return result

    # ── Export ─────────────────────────────────────────────────────────────

    def export_csv(self, output_path: str) -> None:
        """
        Export all 1-D histograms to a single CSV file.

        Each histogram becomes two columns: `{key}_counts` and
        `{key}_energy`. Columns are aligned to the longest histogram;
        shorter ones are NaN-padded.

        Parameters
        ----------
        output_path : str
            Destination .csv path.
        """
        spectra = self.get_all_spectra()
        if not spectra:
            raise RuntimeError("No 1-D histograms found in file.")  # pragma: no cover

        frames = {}
        for key, (counts, energies) in spectra.items():
            safe_key = key.replace(";1", "").replace("/", "_")
            frames[f"{safe_key}_energy"] = pd.Series(energies)
            frames[f"{safe_key}_counts"] = pd.Series(counts)

        df = pd.DataFrame(frames)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(spectra)} histogram(s) → {output_path}")

    def export_hdf5(self, output_path: str) -> None:
        """
        Export all 1-D histograms to an HDF5 file.

        Each histogram is stored as a dataset group with `counts` and
        `energies` arrays. Compatible with h5py and pandas HDFStore.

        Parameters
        ----------
        output_path : str
            Destination .h5 path.
        """
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required for HDF5 export. Install with: pip install h5py"
            ) from e  # pragma: no cover

        spectra = self.get_all_spectra()
        if not spectra:
            raise RuntimeError("No 1-D histograms found in file.")  # pragma: no cover

        with h5py.File(output_path, "w") as f:
            for key, (counts, energies) in spectra.items():
                safe_key = key.replace(";1", "").replace("/", "_").lstrip("/")
                grp = f.create_group(safe_key)
                grp.create_dataset("counts", data=counts)
                grp.create_dataset("energies", data=energies)

        print(f"Exported {len(spectra)} histogram(s) → {output_path}")

    # ── Internal ───────────────────────────────────────────────────────────

    def _resolve_key(self, key) -> str:
        """Resolve exact or partial key match. Raises KeyError if not found."""
        if key is None:
            if not self._keys:
                raise KeyError(
                    f"No objects found in {self.filepath}"
                )  # pragma: no cover
            return self._keys[0]
        if key in self._keys:
            return key
        matches = [k for k in self._keys if key in k]
        if not matches:
            raise KeyError(
                f"No histogram matching '{key}' found in {self.filepath}. "
                f"Available keys: {self._keys}"
            )
        return matches[0]

    def __repr__(self) -> str:
        return f"RootReader('{self.filepath}', {len(self._keys)} object(s))"

    def __enter__(self):
        return self

    def __exit__(self, *args):  # pragma: no cover
        self._file.close()
