"""
triples_sigfast.cli.welcome
────────────────────────────
CERN ROOT-inspired terminal welcome page for triples-sigfast.

Displayed when:
  1. User runs `sigfast` with no arguments
  2. User runs `sigfast welcome`
  3. User runs `python -c "import triples_sigfast; triples_sigfast.welcome()"`
"""

from __future__ import annotations

import random
import sys
import time

from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()


# ── Colour palette ────────────────────────────────────────────────────────────
_ACCENT = "bright_cyan"
_DIM = "grey50"
_GOLD = "yellow"
_GREEN = "bright_green"
_RED = "bright_red"
_BLUE = "steel_blue1"
_MAGENTA = "medium_orchid"
_WHITE = "bright_white"


# ── ASCII art logo ─────────────────────────────────────────────────────────────
# Inspired by CERN ROOT's startup art — bold, minimal, physics-coded.

_LOGO = r"""
  ████████╗██████╗ ██╗██████╗ ██╗     ███████╗███████╗
     ██╔══╝██╔══██╗██║██╔══██╗██║     ██╔════╝██╔════╝
     ██║   ██████╔╝██║██████╔╝██║     █████╗  ███████╗
     ██║   ██╔══██╗██║██╔═══╝ ██║     ██╔══╝  ╚════██║
     ██║   ██║  ██║██║██║     ███████╗███████╗███████║
     ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝     ╚══════╝╚══════╝╚══════╝

        ███████╗██╗ ██████╗ ███████╗ █████╗ ███████╗████████╗
        ██╔════╝██║██╔════╝ ██╔════╝██╔══██╗██╔════╝╚══██╔══╝
        ███████╗██║██║  ███╗█████╗  ███████║███████╗   ██║
        ╚════██║██║██║   ██║██╔══╝  ██╔══██║╚════██║   ██║
        ███████║██║╚██████╔╝██║     ██║  ██║███████║   ██║
        ╚══════╝╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚══════╝   ╚═╝
"""

# Compact single-line logo for narrow terminals
_LOGO_COMPACT = "  ⚡  T R I P L E S - S I G F A S T  ⚡"


# ── Physics quote bank ────────────────────────────────────────────────────────
_QUOTES = [
    (
        "Not only is the Universe stranger than we think,\n"
        " it is stranger than we can think.",
        "Werner Heisenberg",
    ),
    (
        "The most incomprehensible thing about the world\n"
        " is that it is comprehensible.",
        "Albert Einstein",
    ),
    (
        "Anyone who is not shocked by quantum mechanics\n has not understood it.",
        "Niels Bohr",
    ),
    (
        "Physics is the only real science.\n The rest are just stamp collecting.",
        "Ernest Rutherford",
    ),
    ("The electron is not as simple as it looks.", "W. Lawrence Bragg"),
    ("God does not play dice with the Universe.", "Albert Einstein"),
    (
        "The good thing about science is that it's true\n"
        " whether or not you believe in it.",
        "Neil deGrasse Tyson",
    ),
    (
        "Energy cannot be created or destroyed,\n"
        " it can only be changed from one form to another.",
        "Albert Einstein",
    ),
    (
        "In physics, you don't have to go around making trouble\n"
        " for yourself — nature does it for you.",
        "Frank Wilczek",
    ),
    ("The laws of physics are the same everywhere in the universe.", "Emmy Noether"),
]


def _get_version() -> str:
    try:
        import triples_sigfast

        return triples_sigfast.__version__
    except Exception:
        return "1.7.0"


def _draw_logo(wide: bool) -> None:
    """Render the ASCII art logo with gradient-style colouring."""
    if wide:
        lines = _LOGO.splitlines()
        colours = [
            _ACCENT,
            _BLUE,
            _BLUE,
            _ACCENT,
            _BLUE,
            _BLUE,
            "",
            _MAGENTA,
            _MAGENTA,
            _ACCENT,
            _MAGENTA,
            _MAGENTA,
            _MAGENTA,
        ]
        for line, colour in zip(lines, colours + [_DIM] * 20):
            if colour:
                console.print(Align.center(f"[{colour}]{line}[/{colour}]"))
            else:
                console.print()
    else:
        console.print()
        console.print(Align.center(f"[bold {_ACCENT}]{_LOGO_COMPACT}[/bold {_ACCENT}]"))
        console.print()


def _draw_tagline(version: str) -> None:
    """Version + tagline bar."""
    tag = Text()
    tag.append("  v", style=_DIM)
    tag.append(version, style=f"bold {_GOLD}")
    tag.append("  │  ", style=_DIM)
    tag.append(
        "JIT-compiled  ·  Nuclear Physics  ·  High-Energy Physics  ·  Signal Processing",
        style=_DIM,
    )
    tag.append("  ", style=_DIM)
    console.print(Align.center(tag))


def _draw_feature_cards() -> None:
    """Three side-by-side feature cards, inspired by ROOT's module grid."""

    nuclear_lines = [
        f"[{_GREEN}]●[/{_GREEN}] ICRP-74 Dose Conversion",
        f"[{_GREEN}]●[/{_GREEN}] ANSI/ANS-6.4.3 Shielding",
        f"[{_GREEN}]●[/{_GREEN}] Geant4 / FLUKA / MCNP / SERPENT",
        f"[{_GREEN}]●[/{_GREEN}] Watt Fission Spectrum",
        f"[{_GREEN}]●[/{_GREEN}] NUBASE2020 Isotope Database",
    ]

    hep_lines = [
        f"[{_ACCENT}]●[/{_ACCENT}] LHE + HepMC3 Event Readers",
        f"[{_ACCENT}]●[/{_ACCENT}] @njit Lorentz Kinematics",
        f"[{_ACCENT}]●[/{_ACCENT}] Invariant Mass  ·  η  ·  ΔR  ·  pT",
        f"[{_ACCENT}]●[/{_ACCENT}] Anti-kT Jet Clustering",
        f"[{_ACCENT}]●[/{_ACCENT}] PYTHIA / Herwig / MadGraph I/O",
    ]

    signal_lines = [
        f"[{_MAGENTA}]●[/{_MAGENTA}] Savitzky-Golay Smoothing",
        f"[{_MAGENTA}]●[/{_MAGENTA}] EMA + Rolling Average",
        f"[{_MAGENTA}]●[/{_MAGENTA}] Peak Detection (Numba)",
        f"[{_MAGENTA}]●[/{_MAGENTA}] Monte Carlo Convergence",
        f"[{_MAGENTA}]●[/{_MAGENTA}] PDF AutoReport Generator",
    ]

    def _card(title: str, colour: str, lines: list[str]) -> Panel:
        body = "\n".join(lines)
        return Panel(
            body,
            title=f"[bold {colour}]{title}[/bold {colour}]",
            border_style=colour,
            padding=(0, 2),
            expand=True,
        )

    cards = Columns(
        [
            _card("☢  Nuclear Physics", _GREEN, nuclear_lines),
            _card("⚛  High-Energy Physics", _ACCENT, hep_lines),
            _card("📡  Signal & Stats", _MAGENTA, signal_lines),
        ],
        equal=True,
        expand=True,
    )
    console.print(cards)


def _draw_performance_bar() -> None:
    """Compact performance summary row."""
    table = Table.grid(expand=True, padding=(0, 3))
    table.add_column(justify="center")
    table.add_column(justify="center")
    table.add_column(justify="center")
    table.add_column(justify="center")

    def _stat(value: str, label: str, colour: str) -> Text:
        t = Text()
        t.append(value, style=f"bold {colour}")
        t.append(f"\n{label}", style=_DIM)
        return t

    table.add_row(
        _stat("@njit + prange", "Parallel JIT Backend", _GOLD),
        _stat("100M+ rows/s", "Throughput", _GREEN),
        _stat("470 tests", "Test Suite", _ACCENT),
        _stat("≥ 15× faster", "vs. readline on 2 GB LHE", _MAGENTA),
    )
    console.print(
        Panel(
            table,
            border_style=_DIM,
            padding=(0, 1),
            title=f"[{_DIM}]Performance[/{_DIM}]",
        )
    )


def _draw_quick_start() -> None:
    """Quick-start commands block."""
    console.print(f"\n[bold {_WHITE}]Quick Start[/bold {_WHITE}]")
    console.print(Rule(style=_DIM))

    cmds = [
        (
            "analyze",
            "sigfast analyze simulation.root",
            "Read + smooth + peak-detect a Geant4/MCNP file",
        ),
        (
            "dose",
            "sigfast dose --flux 1e6 --energy 2.35 --particle neutron",
            "ICRP-74 dose rate calculation",
        ),
        (
            "shield",
            "sigfast shield --material lead --thickness 10 --energy 1.25",
            "ANSI/ANS-6.4.3 shielding transmission",
        ),
        (
            "HEP",
            "from triples_sigfast.hep.jets import cluster_jets",
            "Anti-kT jet clustering in pure Python",
        ),
        (
            "LHE",
            "from triples_sigfast.io import LHEReader",
            "Block-memory PYTHIA event file reader",
        ),
    ]

    for label, cmd, desc in cmds:
        label_text = Text(f"  [{label:<8}]", style=f"bold {_GOLD}")
        cmd_text = Text(cmd, style=f"bold {_ACCENT}")
        desc_text = Text(f"  # {desc}", style=_DIM)
        line = Text.assemble(label_text, "  ", cmd_text, desc_text)
        console.print(line)


def _draw_links() -> None:
    """Footer with links."""
    console.print(Rule(style=_DIM))
    table = Table.grid(expand=True, padding=(0, 4))
    table.add_column(justify="left")
    table.add_column(justify="center")
    table.add_column(justify="right")

    table.add_row(
        f"[{_DIM}]📦  pip install triples-sigfast[/{_DIM}]",
        f"[{_DIM}]GitHub  github.com/SamdaniSayam/triples-sigfast[/{_DIM}]",
        f"[{_DIM}]PyPI  pypi.org/project/triples-sigfast[/{_DIM}]",
    )
    console.print(table)


def _draw_quote() -> None:
    """Random physics quote footer."""
    quote, author = random.choice(_QUOTES)
    q = Text()
    q.append(f'  "{quote}"\n', style=f"italic {_DIM}")
    q.append(f"    — {author}", style=f"bold {_DIM}")
    console.print(Align.center(q))
    console.print()


def print_welcome(animated: bool = True, _console: Console | None = None) -> None:
    """
    Print the full triples-sigfast welcome page to the terminal.

    Parameters
    ----------
    animated : bool
        If True, renders with a brief fade-in delay (default for TTY).
        Pass False to suppress delay (useful in CI / non-interactive shells).
    _console : Console, optional
        Internal — allows injecting a recording console for SVG export.

    Examples
    --------
    >>> from triples_sigfast.cli.welcome import print_welcome
    >>> print_welcome()
    """
    global console
    if _console is not None:
        console = _console

    # Skip animation in non-interactive environments
    if not sys.stdout.isatty():
        animated = False

    wide = console.width >= 120

    console.print()
    _draw_logo(wide)

    if animated:
        time.sleep(0.05)

    version = _get_version()
    _draw_tagline(version)
    console.print()

    if animated:
        time.sleep(0.05)

    _draw_feature_cards()
    console.print()

    if animated:
        time.sleep(0.03)

    _draw_performance_bar()
    _draw_quick_start()
    _draw_links()

    if animated:
        time.sleep(0.02)

    _draw_quote()
