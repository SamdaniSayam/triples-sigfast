"""
triples_sigfast.io
------------------
Simulation and raw-data file reader package.

This sub-package provides backend reader classes for every simulation output
format supported by triples-sigfast, as well as a universal dispatcher
(SimReader) that selects the correct backend automatically based on the file
extension.

Public API
----------
SimReader   -- Universal dispatcher; the recommended entry point for all I/O.
RootReader  -- Geant4 ROOT output files (.root) via uproot.
FlukaReader -- FLUKA output files (.flair, .lis).
MCNPReader  -- MCNP output files (.mctal).
SerpentReader -- SERPENT output files (.det, .m).
LHEReader   -- Les Houches Event files (.lhe) for PYTHIA/MadGraph events.
HepMCReader -- HepMC3 event files (.hepmc, .hepmc3) for PYTHIA/Herwig events.
RawReader   -- Plain-text columnar data files (.csv, .tsv, .txt, .dat, .asc, .out).

Usage
-----
The preferred approach is always to use SimReader, which selects the backend
automatically:

    from triples_sigfast.io import SimReader
    reader = SimReader("output.root")      # Geant4
    reader = SimReader("spectrum.csv")     # plain-text CSV
    counts, energies = reader.get_spectrum()

Backends can also be instantiated directly for advanced use cases.
"""

from .fluka import FlukaReader
from .hepmc import HepMCReader
from .lhe import LHEReader
from .mcnp import MCNPReader
from .raw import RawReader
from .root_reader import RootReader
from .serpent import SerpentReader
from .sim_reader import SimReader

__all__ = [
    "SimReader",
    "RootReader",
    "FlukaReader",
    "MCNPReader",
    "SerpentReader",
    "LHEReader",
    "HepMCReader",
    "RawReader",
]
