"""
triples_sigfast.cli.welcome
----------------------------
Terminal welcome page for triples-sigfast.

Displayed automatically when a user runs `sigfast` with no subcommand, or
explicitly with `sigfast welcome`.  The layout is inspired by the CERN ROOT
framework's startup screen: a large ASCII logo, a feature grid, a performance
summary panel, a quick-start command reference, and a closing physics quote.

The page degrades gracefully on narrow terminals (< 120 columns) by replacing
the full two-row ASCII art with a compact single-line header.

Rendering sequence
------------------
1. _draw_logo()            -- ASCII art banner (wide) or compact header (narrow)
2. _draw_tagline()         -- version string and one-line feature summary
3. _draw_feature_cards()   -- three-column module feature grid
4. _draw_performance_bar() -- key performance statistics panel
5. _draw_quick_start()     -- copy-paste command reference block
6. _draw_links()           -- pip / GitHub / PyPI footer links
7. _draw_quote()           -- randomly selected physics quote
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

# Module-level console instance.  Can be overridden by callers who need to
# redirect output (e.g., for SVG export during documentation builds).
console = Console()


# ---------------------------------------------------------------------------
# Colour palette
# All colour names are Rich style tokens.  Centralised here so the entire
# page can be recoloured by changing these constants.
# ---------------------------------------------------------------------------
_ACCENT  = "bright_cyan"    # primary accent -- headings and bullet markers
_DIM     = "grey50"         # secondary text -- labels, dim annotations
_GOLD    = "yellow"         # highlight -- version number, command labels
_GREEN   = "bright_green"   # nuclear physics card
_BLUE    = "steel_blue1"    # logo gradient, alternate accent
_MAGENTA = "medium_orchid"  # signal/stats card
_WHITE   = "bright_white"   # section headings


# ---------------------------------------------------------------------------
# ASCII art logo
# ---------------------------------------------------------------------------
# Two-row block-letter logo for wide terminals (>= 120 columns).
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

# Compact single-line header for narrow terminals (< 120 columns).
_LOGO_COMPACT = "  T R I P L E S - S I G F A S T"


# ---------------------------------------------------------------------------
# Physics quote bank
# One quote is selected at random for each welcome page render.
# ---------------------------------------------------------------------------
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
        " for yourself -- nature does it for you.",
        "Frank Wilczek",
    ),
    ("The laws of physics are the same everywhere in the universe.", "Emmy Noether"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_version() -> str:
    """Return the installed package version string.

    Falls back to the hard-coded value if the package metadata is unavailable
    (e.g., in a development checkout before installation).
    """
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "1.7.0"


def _draw_logo(wide: bool) -> None:
    """Render the ASCII art logo centred in the terminal.

    Parameters
    ----------
    wide : bool
        If True, render the full two-row block-letter logo.
        If False, render the compact single-line header instead.
    """
    if wide:
        # Apply a gradient-style colouring: alternate accent and blue shades
        # across the two rows of the logo to give a layered visual effect.
        lines = _LOGO.splitlines()
        colours = [
            _ACCENT, _BLUE, _BLUE, _ACCENT, _BLUE, _BLUE, "",
            _MAGENTA, _MAGENTA, _ACCENT, _MAGENTA, _MAGENTA, _MAGENTA,
        ]
        for line, colour in zip(lines, colours + [_DIM] * 20):
            if colour:
                console.print(Align.center(f"[{colour}]{line}[/{colour}]"))
            else:
                console.print()
    else:
        # Narrow terminal: use the compact single-line header.
        console.print()
        console.print(Align.center(f"[bold {_ACCENT}]{_LOGO_COMPACT}[/bold {_ACCENT}]"))
        console.print()


def _draw_tagline(version: str) -> None:
    """Render the version string and one-line feature tagline below the logo."""
    tag = Text()
    tag.append("  v", style=_DIM)
    tag.append(version, style=f"bold {_GOLD}")
    tag.append("  |  ", style=_DIM)
    tag.append(
        "JIT-compiled  |  Nuclear Physics  |  High-Energy Physics  |  Signal Processing",
        style=_DIM,
    )
    tag.append("  ", style=_DIM)
    console.print(Align.center(tag))


def _draw_feature_cards() -> None:
    """Render three side-by-side feature panels (nuclear, HEP, signal/stats).

    Each panel lists the key capabilities of that sub-package.  The three-
    column layout is similar to the ROOT framework's module grid.
    """

    def _bullet(colour: str, text: str) -> str:
        """Return a Rich markup string for a coloured bullet-point line."""
        return f"[{colour}]*[/{colour}] {text}"

    # Nuclear physics feature list
    nuclear_lines = [
        _bullet(_GREEN, "ICRP-74 Dose Conversion"),
        _bullet(_GREEN, "ANSI/ANS-6.4.3 Shielding"),
        _bullet(_GREEN, "Geant4 / FLUKA / MCNP / SERPENT"),
        _bullet(_GREEN, "Watt Fission Spectrum"),
        _bullet(_GREEN, "NUBASE2020 Isotope Database"),
    ]

    # High-energy physics feature list
    hep_lines = [
        _bullet(_ACCENT, "LHE + HepMC3 Event Readers"),
        _bullet(_ACCENT, "@njit Lorentz Kinematics"),
        _bullet(_ACCENT, "Invariant Mass  |  eta  |  dR  |  pT"),
        _bullet(_ACCENT, "Anti-kT Jet Clustering"),
        _bullet(_ACCENT, "PYTHIA / Herwig / MadGraph I/O"),
    ]

    # Signal processing and statistics feature list
    signal_lines = [
        _bullet(_MAGENTA, "Savitzky-Golay Smoothing"),
        _bullet(_MAGENTA, "EMA + Rolling Average"),
        _bullet(_MAGENTA, "Peak Detection (Numba JIT)"),
        _bullet(_MAGENTA, "Monte Carlo Convergence"),
        _bullet(_MAGENTA, "PDF AutoReport Generator"),
    ]

    def _card(title: str, colour: str, lines: list[str]) -> Panel:
        """Build a Rich Panel for one feature column."""
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
            _card("Nuclear Physics",      _GREEN,   nuclear_lines),
            _card("High-Energy Physics",  _ACCENT,  hep_lines),
            _card("Signal and Stats",     _MAGENTA, signal_lines),
        ],
        equal=True,
        expand=True,
    )
    console.print(cards)


def _draw_performance_bar() -> None:
    """Render a compact four-column performance statistics panel."""
    table = Table.grid(expand=True, padding=(0, 3))
    table.add_column(justify="center")
    table.add_column(justify="center")
    table.add_column(justify="center")
    table.add_column(justify="center")

    def _stat(value: str, label: str, colour: str) -> Text:
        """Build a two-line Text object: bold value on line 1, dim label on line 2."""
        t = Text()
        t.append(value, style=f"bold {colour}")
        t.append(f"\n{label}", style=_DIM)
        return t

    table.add_row(
        _stat("@njit + prange",  "Parallel JIT Backend",        _GOLD),
        _stat("100M+ rows/s",    "Throughput",                   _GREEN),
        _stat("470 tests",       "Test Suite",                   _ACCENT),
        _stat(">= 15x faster",   "vs. readline on 2 GB LHE",    _MAGENTA),
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
    """Render a copy-paste command reference block below the performance panel."""
    console.print(f"\n[bold {_WHITE}]Quick Start[/bold {_WHITE}]")
    console.print(Rule(style=_DIM))

    # Each entry: (short label, command string, explanatory comment)
    commands = [
        (
            "analyze",
            "sigfast analyze simulation.root",
            "Read, smooth, and peak-detect a Geant4/MCNP file",
        ),
        (
            "dose",
            "sigfast dose --flux 1e6 --energy 2.35 --particle neutron",
            "ICRP-74 ambient dose rate calculation",
        ),
        (
            "shield",
            "sigfast shield --material lead --thickness 10 --energy 1.25",
            "ANSI/ANS-6.4.3 gamma transmission calculation",
        ),
        (
            "HEP",
            "from triples_sigfast.hep.jets import cluster_jets",
            "Anti-kT jet clustering in pure Python",
        ),
        (
            "LHE",
            "from triples_sigfast.io import LHEReader",
            "Block-buffered PYTHIA event file reader",
        ),
    ]

    for label, cmd, desc in commands:
        label_text = Text(f"  [{label:<8}]", style=f"bold {_GOLD}")
        cmd_text   = Text(cmd,               style=f"bold {_ACCENT}")
        desc_text  = Text(f"  # {desc}",     style=_DIM)
        line = Text.assemble(label_text, "  ", cmd_text, desc_text)
        console.print(line)


def _draw_links() -> None:
    """Render a footer row with pip install, GitHub, and PyPI links."""
    console.print(Rule(style=_DIM))
    table = Table.grid(expand=True, padding=(0, 4))
    table.add_column(justify="left")
    table.add_column(justify="center")
    table.add_column(justify="right")

    table.add_row(
        f"[{_DIM}]pip install triples-sigfast[/{_DIM}]",
        f"[{_DIM}]GitHub  github.com/SamdaniSayam/triples-sigfast[/{_DIM}]",
        f"[{_DIM}]PyPI  pypi.org/project/triples-sigfast[/{_DIM}]",
    )
    console.print(table)


def _draw_quote() -> None:
    """Render a randomly selected physics quote centred in the terminal."""
    quote, author = random.choice(_QUOTES)
    q = Text()
    q.append(f'  "{quote}"\n', style=f"italic {_DIM}")
    q.append(f"    -- {author}", style=f"bold {_DIM}")
    console.print(Align.center(q))
    console.print()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def print_welcome(animated: bool = True, _console: Console | None = None) -> None:
    """Print the full triples-sigfast welcome page to the terminal.

    Parameters
    ----------
    animated : bool
        When True, inserts brief pauses between sections to give a fade-in
        effect (default for interactive TTY sessions).  Set to False to
        suppress all delays in CI environments or non-interactive shells.
    _console : Console, optional
        Allows an external caller to inject a recording console for SVG or
        HTML export.  Not intended for normal use.

    Examples
    --------
    >>> from triples_sigfast.cli.welcome import print_welcome
    >>> print_welcome(animated=False)
    """
    global console
    if _console is not None:
        console = _console

    # Disable animation automatically in non-interactive environments
    # (e.g., piped output, CI pipelines, script execution).
    if not sys.stdout.isatty():
        animated = False

    # Determine whether the terminal is wide enough for the full logo.
    wide = console.width >= 120

    # Render each section in sequence, with optional brief delays between
    # them to produce a subtle fade-in effect on interactive terminals.
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
