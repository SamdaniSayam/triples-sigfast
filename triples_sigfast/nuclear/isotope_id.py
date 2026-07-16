"""
triples_sigfast.nuclear.isotope_id
-----------------------------------
JIT-compiled isotope identification from gamma-ray peak energies.

Given an array of detected peak energies (in MeV) from a gamma spectrum,
identifies the most likely radioactive isotopes present using a transpiled
decision tree compiled with Numba ``@njit``.

Architecture
------------
The core matcher is a **pure if/else decision tree** wrapped in
``@njit(cache=True)``.  No scikit-learn, XGBoost, or any ML library is
used at runtime.  The tree is pre-baked into the source and compiles
directly to LLVM via Numba.

Isotope codes (integers) are mapped back to human-readable names in the
Python wrapper functions ``identify_isotopes`` and ``isotope_summary``.

Isotope Signature Database
--------------------------
Energies in MeV — sourced from NNDC / IAEA Nuclear Data Services:

== ============= ============================================
ID Isotope       Gamma energies (MeV)
== ============= ============================================
 0 Cs-137        0.662
 1 Co-60         1.173, 1.332
 2 Na-22         0.511, 1.275
 3 Am-241        0.0595
 4 Ba-133        0.081, 0.276, 0.303, 0.356, 0.384
 5 Eu-152        0.122, 0.245, 0.344, 0.779, 0.964, 1.112, 1.408
 6 K-40          1.461
 7 I-131         0.284, 0.364, 0.637
 8 Mn-54         0.835
 9 Zn-65         1.116
10 Fe-59         1.099, 1.292
11 Annihilation  0.511
12 Tl-208        2.614
13 Bi-214        0.609, 1.120, 1.764
== ============= ============================================
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "identify_isotopes",
    "isotope_summary",
]

# ---------------------------------------------------------------------------
# Isotope code ↔ name mapping
# ---------------------------------------------------------------------------

_ISOTOPE_NAMES: dict[int, str] = {
    0: "Cs-137",
    1: "Co-60",
    2: "Na-22",
    3: "Am-241",
    4: "Ba-133",
    5: "Eu-152",
    6: "K-40",
    7: "I-131",
    8: "Mn-54",
    9: "Zn-65",
    10: "Fe-59",
    11: "Annihilation",
    12: "Tl-208",
    13: "Bi-214",
}

# Reference energies used in the decision tree (MeV), indexed by isotope code.
_ISOTOPE_ENERGIES: dict[int, list[float]] = {
    0: [0.662],
    1: [1.173, 1.332],
    2: [0.511, 1.275],
    3: [0.0595],
    4: [0.081, 0.276, 0.303, 0.356, 0.384],
    5: [0.122, 0.245, 0.344, 0.779, 0.964, 1.112, 1.408],
    6: [1.461],
    7: [0.284, 0.364, 0.637],
    8: [0.835],
    9: [1.116],
    10: [1.099, 1.292],
    11: [0.511],
    12: [2.614],
    13: [0.609, 1.120, 1.764],
}

# Flat lookup: (isotope_code, reference_energy_mev) for every known line.
# Built once at import time, used by the Python wrapper to report the
# matched reference energy.
_ALL_LINES: list[tuple[int, float]] = []
for _code, _energies in _ISOTOPE_ENERGIES.items():
    for _e in _energies:
        _ALL_LINES.append((_code, _e))
_ALL_LINES.sort(key=lambda x: x[1])


# ---------------------------------------------------------------------------
# JIT-compiled decision tree
# ---------------------------------------------------------------------------
# The tree is organised as a binary search on energy to keep depth ≤ 7.
# Each leaf checks ``abs(energy - ref) < tol``.
#
# Sorted unique reference energies:
#   0.0595  0.081  0.122  0.245  0.276  0.284  0.303  0.344  0.356
#   0.364   0.384  0.511  0.609  0.637  0.662  0.779  0.835  0.964
#   1.099   1.112  1.116  1.120  1.173  1.275  1.292  1.332  1.408
#   1.461   1.764  2.614
#
# The function returns a *pair* (isotope_code, line_index) packed into a
# single int64 as ``isotope_code * 100 + line_index``.  The wrapper
# unpacks this.  A return value of -1 means no match.


@njit(cache=True)
def _match_peak(energy: float, tolerance: float) -> int:  # noqa: C901
    """Return an encoded match ``isotope_code * 100 + line_idx``, or -1.

    Parameters
    ----------
    energy : float
        Peak energy in MeV.
    tolerance : float
        Matching window in MeV (symmetric).

    Returns
    -------
    int
        Encoded match or -1 if nothing matches.
    """
    # ── depth 1: split at ~0.65 MeV ──────────────────────────────
    if energy < 0.65:
        # ── depth 2: split at ~0.35 MeV ─────────────────────────
        if energy < 0.35:
            # ── depth 3: split at ~0.25 MeV ──────────────────────
            if energy < 0.25:
                # ── depth 4: split at ~0.10 MeV ──────────────────
                if energy < 0.10:
                    # Am-241 0.0595 | Ba-133 0.081
                    if abs(energy - 0.0595) < tolerance:
                        return 3 * 100 + 0  # Am-241, line 0
                    if abs(energy - 0.081) < tolerance:
                        return 4 * 100 + 0  # Ba-133, line 0
                    return -1
                else:
                    # Eu-152 0.122 | Eu-152 0.245
                    if abs(energy - 0.122) < tolerance:
                        return 5 * 100 + 0  # Eu-152, line 0
                    if abs(energy - 0.245) < tolerance:
                        return 5 * 100 + 1  # Eu-152, line 1
                    return -1
            else:
                # 0.25 ≤ energy < 0.35
                # Ba-133 0.276 | I-131 0.284 | Ba-133 0.303 | Eu-152 0.344
                if energy < 0.30:
                    if abs(energy - 0.276) < tolerance:
                        return 4 * 100 + 1  # Ba-133, line 1
                    if abs(energy - 0.284) < tolerance:
                        return 7 * 100 + 0  # I-131, line 0
                    return -1
                else:
                    if abs(energy - 0.303) < tolerance:
                        return 4 * 100 + 2  # Ba-133, line 2
                    if abs(energy - 0.344) < tolerance:
                        return 5 * 100 + 2  # Eu-152, line 2
                    return -1
        else:
            # 0.35 ≤ energy < 0.65
            # ── depth 3: split at ~0.50 MeV ──────────────────────
            if energy < 0.50:
                # Ba-133 0.356 | I-131 0.364 | Ba-133 0.384
                if abs(energy - 0.356) < tolerance:
                    return 4 * 100 + 3  # Ba-133, line 3
                if abs(energy - 0.364) < tolerance:
                    return 7 * 100 + 1  # I-131, line 1
                if abs(energy - 0.384) < tolerance:
                    return 4 * 100 + 4  # Ba-133, line 4
                return -1
            else:
                # 0.50 ≤ energy < 0.65
                # Na-22 0.511 / Annihilation 0.511 | Bi-214 0.609 | I-131 0.637
                if energy < 0.56:
                    # Na-22 and Annihilation share 0.511
                    # Return Na-22 first (more specific); caller can
                    # also check Annihilation via a second pass.
                    if abs(energy - 0.511) < tolerance:
                        return 2 * 100 + 0  # Na-22, line 0 (0.511)
                    return -1
                else:
                    if abs(energy - 0.609) < tolerance:
                        return 13 * 100 + 0  # Bi-214, line 0
                    if abs(energy - 0.637) < tolerance:
                        return 7 * 100 + 2  # I-131, line 2
                    return -1
    else:
        # energy ≥ 0.65
        # ── depth 2: split at ~1.15 MeV ─────────────────────────
        if energy < 1.15:
            # ── depth 3: split at ~0.90 MeV ──────────────────────
            if energy < 0.90:
                # Cs-137 0.662 | Eu-152 0.779 | Mn-54 0.835
                if abs(energy - 0.662) < tolerance:
                    return 0 * 100 + 0  # Cs-137, line 0
                if abs(energy - 0.779) < tolerance:
                    return 5 * 100 + 3  # Eu-152, line 3
                if abs(energy - 0.835) < tolerance:
                    return 8 * 100 + 0  # Mn-54, line 0
                return -1
            else:
                # 0.90 ≤ energy < 1.15
                # Eu-152 0.964 | Fe-59 1.099 | Eu-152 1.112 |
                # Zn-65 1.116 | Bi-214 1.120
                if energy < 1.05:
                    if abs(energy - 0.964) < tolerance:
                        return 5 * 100 + 4  # Eu-152, line 4
                    return -1
                else:
                    # Cluster: 1.099, 1.112, 1.116, 1.120
                    if abs(energy - 1.099) < tolerance:
                        return 10 * 100 + 0  # Fe-59, line 0
                    if abs(energy - 1.112) < tolerance:
                        return 5 * 100 + 5  # Eu-152, line 5
                    if abs(energy - 1.116) < tolerance:
                        return 9 * 100 + 0  # Zn-65, line 0
                    if abs(energy - 1.120) < tolerance:
                        return 13 * 100 + 1  # Bi-214, line 1
                    return -1
        else:
            # energy ≥ 1.15
            # ── depth 3: split at ~1.40 MeV ──────────────────────
            if energy < 1.40:
                # Co-60 1.173 | Na-22 1.275 | Fe-59 1.292 | Co-60 1.332
                if energy < 1.25:
                    if abs(energy - 1.173) < tolerance:
                        return 1 * 100 + 0  # Co-60, line 0
                    return -1
                else:
                    if abs(energy - 1.275) < tolerance:
                        return 2 * 100 + 1  # Na-22, line 1
                    if abs(energy - 1.292) < tolerance:
                        return 10 * 100 + 1  # Fe-59, line 1
                    if abs(energy - 1.332) < tolerance:
                        return 1 * 100 + 1  # Co-60, line 1
                    return -1
            else:
                # energy ≥ 1.40
                # Eu-152 1.408 | K-40 1.461 | Bi-214 1.764 | Tl-208 2.614
                if energy < 1.60:
                    if abs(energy - 1.408) < tolerance:
                        return 5 * 100 + 6  # Eu-152, line 6
                    if abs(energy - 1.461) < tolerance:
                        return 6 * 100 + 0  # K-40, line 0
                    return -1
                else:
                    if abs(energy - 1.764) < tolerance:
                        return 13 * 100 + 2  # Bi-214, line 2
                    if abs(energy - 2.614) < tolerance:
                        return 12 * 100 + 0  # Tl-208, line 0
                    return -1


# ---------------------------------------------------------------------------
# Secondary JIT pass for 0.511 MeV ambiguity (Na-22 vs Annihilation)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _match_annihilation(energy: float, tolerance: float) -> int:
    """Check if *energy* matches the 0.511 MeV annihilation line.

    Returns ``11 * 100 + 0`` (Annihilation, line 0) on match, else -1.
    """
    if abs(energy - 0.511) < tolerance:
        return 11 * 100 + 0
    return -1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def identify_isotopes(
    peak_energies: np.ndarray,
    tolerance_kev: float = 2.0,
) -> list[dict]:
    """Identify isotopes from detected gamma-ray peak energies.

    Parameters
    ----------
    peak_energies : np.ndarray
        1-D array of peak energies in **MeV**.
    tolerance_kev : float, optional
        Matching tolerance in **keV** (default 2.0 keV).

    Returns
    -------
    list[dict]
        Each dict contains:

        - ``'isotope'``  — isotope name (e.g. ``'Co-60'``).
        - ``'energy_mev'`` — reference line energy in MeV.
        - ``'matched_peak_mev'`` — the input peak energy that matched.
        - ``'confidence'`` — ``'high'`` if |Δ| ≤ 1 keV, else ``'medium'``.

    Notes
    -----
    If a peak at ~0.511 MeV is detected, both ``Na-22`` **and**
    ``Annihilation`` are reported since the line is shared.  Na-22 is
    only reported if a companion peak at ~1.275 MeV is also present,
    avoiding false positives from standalone annihilation radiation.
    """
    if peak_energies.ndim == 0:
        peak_energies = peak_energies.reshape(1)

    tol_mev = tolerance_kev * 1.0e-3  # keV → MeV
    results: list[dict] = []

    for peak in peak_energies:
        code = _match_peak(float(peak), tol_mev)
        if code >= 0:
            iso_code = code // 100
            line_idx = code % 100
            ref_energy = _ISOTOPE_ENERGIES[iso_code][line_idx]
            delta_kev = abs(float(peak) - ref_energy) * 1.0e3
            confidence = "high" if delta_kev <= 1.0 + 1e-9 else "medium"
            results.append(
                {
                    "isotope": _ISOTOPE_NAMES[iso_code],
                    "energy_mev": ref_energy,
                    "matched_peak_mev": float(peak),
                    "confidence": confidence,
                }
            )

        # Second pass: 0.511 MeV ambiguity
        ann_code = _match_annihilation(float(peak), tol_mev)
        if ann_code >= 0:
            iso_code_a = ann_code // 100
            ref_energy_a = _ISOTOPE_ENERGIES[iso_code_a][0]
            delta_kev_a = abs(float(peak) - ref_energy_a) * 1.0e3
            conf_a = "high" if delta_kev_a <= 1.0 + 1e-9 else "medium"
            results.append(
                {
                    "isotope": _ISOTOPE_NAMES[iso_code_a],
                    "energy_mev": ref_energy_a,
                    "matched_peak_mev": float(peak),
                    "confidence": conf_a,
                }
            )

    # Post-process: remove Na-22 from 0.511 MeV if 1.275 MeV is absent
    na22_at_511 = [
        i
        for i, r in enumerate(results)
        if r["isotope"] == "Na-22" and abs(r["energy_mev"] - 0.511) < 1e-6
    ]
    if na22_at_511:
        has_1275 = any(
            r["isotope"] == "Na-22" and abs(r["energy_mev"] - 1.275) < 1e-6
            for r in results
        )
        if not has_1275:
            for i in sorted(na22_at_511, reverse=True):
                results.pop(i)

    return results


def isotope_summary(
    peak_energies: np.ndarray,
    tolerance_kev: float = 2.0,
) -> str:
    """Return a formatted string summarising identified isotopes.

    Parameters
    ----------
    peak_energies : np.ndarray
        1-D array of peak energies in **MeV**.
    tolerance_kev : float, optional
        Matching tolerance in **keV** (default 2.0 keV).

    Returns
    -------
    str
        Human-readable multi-line summary table.
    """
    matches = identify_isotopes(peak_energies, tolerance_kev=tolerance_kev)

    if not matches:
        return "No isotopes identified."

    header = (
        f"{'Isotope':<15} {'Ref (MeV)':>10} {'Peak (MeV)':>11} "
        f"{'Δ (keV)':>8} {'Confidence':>10}"
    )
    sep = "─" * len(header)
    lines = [sep, header, sep]

    for m in matches:
        delta = abs(m["matched_peak_mev"] - m["energy_mev"]) * 1e3
        lines.append(
            f"{m['isotope']:<15} {m['energy_mev']:>10.4f} "
            f"{m['matched_peak_mev']:>11.4f} {delta:>8.2f} "
            f"{m['confidence']:>10}"
        )

    lines.append(sep)
    lines.append(f"Total matches: {len(matches)}")
    return "\n".join(lines)
