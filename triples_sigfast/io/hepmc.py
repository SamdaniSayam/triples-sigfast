"""
triples_sigfast.io.hepmc
─────────────────────────
High-performance HepMC3 ASCII (Asciiv3) event file reader.

Reads the HepMC3 ASCII format produced by PYTHIA 8, Herwig 7, Sherpa,
and any generator using the HepMC3 library as output backend.

HepMC3 ASCII v3 record types
-----------------------------
E  <id> <n_vertices> <n_weights>          — event record
W  <w1> [w2 ...]                          — weights
U  <momentum_unit> <length_unit>          — units
A  <key> <value>                          — attribute
V  <id> <status> [<n_particles_in>]       — vertex record
P  <id> <vid> <pid> <px> <py> <pz> <e> <m> <status>  — particle

Performance strategy
--------------------
Same block-memory approach as LHEReader: the full file is read in one I/O call,
then split on event boundaries using str.split(). Particle 'P' lines are
accumulated into lists and concatenated once per call to get_particles().

References
----------
HepMC3 paper: https://arxiv.org/abs/1912.08005
Format spec:  https://gitlab.cern.ch/hepmc/HepMC3
"""

from __future__ import annotations

import numpy as np

# ── Column indices for 'P' particle lines ─────────────────────────────────────
#  P  id  vid  pid  px  py  pz  e  m  status  [attributes...]
_P_ID = 1
_P_VID = 2
_P_PID = 3
_P_PX = 4
_P_PY = 5
_P_PZ = 6
_P_E = 7
_P_M = 8
_P_STATUS = 9
_P_MIN_COLS = 10


class HepMCReader:
    """
    Block-memory reader for HepMC3 ASCII (Asciiv3) event files.

    Parses events containing vertex and particle records from files written
    by PYTHIA 8, Herwig 7, Sherpa, or any HepMC3-enabled generator.

    Parameters
    ----------
    filepath : str
        Path to the .hepmc or .hepmc3 file.

    Examples
    --------
    >>> reader = HepMCReader("pythia.hepmc")
    >>> reader.summary()
    >>> particles = reader.get_particles(status=1)  # final-state
    >>> pt = np.sqrt(particles["px"]**2 + particles["py"]**2)
    >>> E_arr, evt_idx = reader.get_spectrum("E")
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._events: list[dict] = []
        self._momentum_unit: str = "GEV"
        self._length_unit: str = "MM"
        self._parse()

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self) -> None:
        """Read the entire file and split into per-event line blocks."""
        with open(self.filepath, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        current_lines: list[str] = []
        in_listing = False

        for raw_line in lines:
            line = raw_line.strip()

            if not line or line.startswith("HepMC::Version"):
                continue
            if "START_EVENT_LISTING" in line:
                in_listing = True
                continue
            if "END_EVENT_LISTING" in line:
                # Flush last event
                if current_lines:
                    evt = self._parse_event_lines(current_lines)
                    if evt is not None:
                        self._events.append(evt)
                break

            if not in_listing:
                continue

            # New event record → flush previous
            if line.startswith("E "):
                if current_lines:
                    evt = self._parse_event_lines(current_lines)
                    if evt is not None:
                        self._events.append(evt)
                current_lines = [line]
            else:
                current_lines.append(line)

        # Handle files without END_EVENT_LISTING marker
        if current_lines:
            evt = self._parse_event_lines(current_lines)
            if evt is not None:
                self._events.append(evt)

    def _parse_event_lines(self, lines: list[str]) -> dict | None:
        """Parse one event's lines into structured arrays."""
        if not lines:
            return None  # pragma: no cover

        event_id: int = 0
        weight: float = 1.0
        pid_list: list[int] = []
        status_list: list[int] = []
        px_list: list[float] = []
        py_list: list[float] = []
        pz_list: list[float] = []
        E_list: list[float] = []
        mass_list: list[float] = []
        particle_id_list: list[int] = []

        for line in lines:
            parts = line.split()
            record_type = parts[0] if parts else ""

            if record_type == "E":
                # E <id> <n_vertices> <n_weights>
                try:
                    event_id = int(parts[1])
                except (IndexError, ValueError):
                    pass  # pragma: no cover

            elif record_type == "W":
                # W <w1> [w2 ...]
                try:
                    weight = float(parts[1])
                except (IndexError, ValueError):
                    pass  # pragma: no cover

            elif record_type == "U":
                # U GEV MM  (units, already set at file level)
                if len(parts) >= 2:
                    self._momentum_unit = parts[1].upper()
                if len(parts) >= 3:
                    self._length_unit = parts[2].upper()

            elif record_type == "P":
                # P id vid pid px py pz e m status [attrs...]
                if len(parts) < _P_MIN_COLS:
                    continue  # pragma: no cover
                try:
                    particle_id_list.append(int(parts[_P_ID]))
                    pid_list.append(int(parts[_P_PID]))
                    px_list.append(float(parts[_P_PX]))
                    py_list.append(float(parts[_P_PY]))
                    pz_list.append(float(parts[_P_PZ]))
                    E_list.append(float(parts[_P_E]))
                    mass_list.append(float(parts[_P_M]))
                    status_list.append(int(parts[_P_STATUS]))
                except (ValueError, IndexError):
                    continue  # pragma: no cover

        if not pid_list:
            return None

        return {
            "event_id": event_id,
            "weight": weight,
            "n_particles": len(pid_list),
            "particle_id": np.array(particle_id_list, dtype=np.int32),
            "pid": np.array(pid_list, dtype=np.int32),
            "status": np.array(status_list, dtype=np.int32),
            "px": np.array(px_list, dtype=np.float64),
            "py": np.array(py_list, dtype=np.float64),
            "pz": np.array(pz_list, dtype=np.float64),
            "E": np.array(E_list, dtype=np.float64),
            "mass": np.array(mass_list, dtype=np.float64),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def get_particles(self, status: int | None = None) -> dict[str, np.ndarray]:
        """
        Return all particle 4-vectors as flat NumPy arrays across all events.

        Parameters
        ----------
        status : int or None
            HepMC3 status filter. Common values:
            1 → final-state (stable),  2 → decayed/fragmented,
            4 → beam particle.
            None (default) returns all particles.

        Returns
        -------
        dict with keys: 'pid', 'status', 'px', 'py', 'pz', 'E', 'mass',
        'event_index' — all float64 except int32 columns.

        Examples
        --------
        >>> p = reader.get_particles(status=1)
        >>> eta = np.arctanh(p["pz"] / np.sqrt(p["px"]**2+p["py"]**2+p["pz"]**2))
        """
        if not self._events:
            raise RuntimeError(f"No events parsed from {self.filepath}")

        pid_chunks: list[np.ndarray] = []
        stat_chunks: list[np.ndarray] = []
        px_chunks: list[np.ndarray] = []
        py_chunks: list[np.ndarray] = []
        pz_chunks: list[np.ndarray] = []
        E_chunks: list[np.ndarray] = []
        mass_chunks: list[np.ndarray] = []
        evt_chunks: list[np.ndarray] = []

        for i, evt in enumerate(self._events):
            if status is not None:
                mask = evt["status"] == status
                if not mask.any():
                    continue
            else:
                mask = np.ones(evt["n_particles"], dtype=bool)

            pid_chunks.append(evt["pid"][mask])
            stat_chunks.append(evt["status"][mask])
            px_chunks.append(evt["px"][mask])
            py_chunks.append(evt["py"][mask])
            pz_chunks.append(evt["pz"][mask])
            E_chunks.append(evt["E"][mask])
            mass_chunks.append(evt["mass"][mask])
            evt_chunks.append(np.full(mask.sum(), i, dtype=np.int32))

        if not pid_chunks:
            empty = np.array([], dtype=np.float64)
            return {
                k: empty
                for k in ("pid", "status", "px", "py", "pz", "E", "mass", "event_index")
            }

        return {
            "pid": np.concatenate(pid_chunks).astype(np.int32),
            "status": np.concatenate(stat_chunks).astype(np.int32),
            "px": np.concatenate(px_chunks),
            "py": np.concatenate(py_chunks),
            "pz": np.concatenate(pz_chunks),
            "E": np.concatenate(E_chunks),
            "mass": np.concatenate(mass_chunks),
            "event_index": np.concatenate(evt_chunks).astype(np.int32),
        }

    def get_spectrum(self, key: str = "E") -> tuple[np.ndarray, np.ndarray]:
        """
        Return (values, event_indices) for a named particle quantity.

        Compatible with sigfast smoothing tools (savitzky_golay, find_peaks, etc.).

        Parameters
        ----------
        key : str
            One of 'E', 'px', 'py', 'pz', 'mass'. Default 'E'.

        Returns
        -------
        values : np.ndarray
        event_indices : np.ndarray
        """
        particles = self.get_particles()
        if key not in particles:
            raise KeyError(
                f"Key '{key}' not available. Choose from: {list(particles.keys())}"
            )
        return particles[key].astype(np.float64), particles["event_index"].astype(
            np.float64
        )

    def get_tally(self, name: str) -> dict:
        """Return a tally-style dict compatible with SimReader.get_tally()."""
        values, bins = self.get_spectrum(name)
        return {
            "name": name,
            "values": values,
            "errors": np.zeros_like(values),
            "bins": bins,
        }

    def keys(self) -> list[str]:
        """Return available particle quantity keys."""
        return ["E", "px", "py", "pz", "mass", "pid", "status", "event_index"]

    def n_events(self) -> int:
        """Return total number of events parsed."""
        return len(self._events)

    def summary(self) -> None:
        """Print a human-readable summary of the HepMC file."""
        print(f"\nHepMC3 file: {self.filepath}")
        print(f"  Momentum unit : {self._momentum_unit}")
        print(f"  Length unit   : {self._length_unit}")
        print(f"  Events parsed : {len(self._events)}")
        if self._events:
            total = sum(e["n_particles"] for e in self._events)
            avg = total / len(self._events)
            weights = np.array([e["weight"] for e in self._events])
            print(f"  Total particles (all status) : {total}")
            print(f"  Avg particles/event          : {avg:.1f}")
            print(
                f"  Weight range                 : [{weights.min():.4e}, {weights.max():.4e}]"
            )
        print()

    def __repr__(self) -> str:
        return f"HepMCReader('{self.filepath}', {len(self._events)} event(s))"

    def __len__(self) -> int:
        return len(self._events)
