"""
tests/test_hep_pdg.py
---------------------
Tests for PDG AOT ingestion.
"""

import numpy as np

from triples_sigfast.hep import pdg


def test_pdg_get_mass():
    # Electron: 11, mass ~ 0.511 MeV = 0.000511 GeV
    m_e = pdg.get_mass(11)
    assert abs(m_e - 0.000511) < 1e-6

    # Z boson: 23, mass ~ 91.1876 GeV
    m_z = pdg.get_mass(23)
    assert abs(m_z - 91.1876) < 0.1


def test_pdg_get_width():
    # Z boson width ~ 2.4952 GeV
    w_z = pdg.get_width(23)
    assert abs(w_z - 2.4952) < 0.1

    # Electron width should be exactly 0 (stable)
    w_e = pdg.get_width(11)
    assert w_e == 0.0


def test_pdg_get_mass_array():
    ids = np.array([11, -11, 23], dtype=np.int64)
    masses = pdg.get_mass_array(ids)

    assert abs(masses[0] - 0.000511) < 1e-6
    assert abs(masses[1] - 0.000511) < 1e-6
    assert abs(masses[2] - 91.1876) < 0.1


def test_pdg_get_width_array():
    ids = np.array([11, 23], dtype=np.int64)
    widths = pdg.get_width_array(ids)

    assert widths[0] == 0.0
    assert abs(widths[1] - 2.4952) < 0.1


def test_unknown_pdgid():
    assert pdg.get_mass(999999999) == 0.0
    assert pdg.get_width(999999999) == 0.0
