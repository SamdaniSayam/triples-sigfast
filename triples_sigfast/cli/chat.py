"""
triples_sigfast.cli.chat
-------------------------
Interactive AI-powered terminal copilot for triples-sigfast.

Provides a REPL-style chat interface backed by Google Gemini that lets
scientists ask questions about their physics simulation data.  When a
simulation file is supplied, the file is analysed automatically and the
results are injected into the system prompt so the model has full context.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown

console = Console()

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are **sigfast copilot**, a nuclear and particle physics analysis assistant
built into the *triples-sigfast* library.

**Library capabilities you can reference:**
- GIL-free, Numba-JIT signal processing (Savitzky-Golay smoothing, peak detection)
- Monte Carlo convergence diagnostics (mean relative error, MCNP-style R < 0.05)
- ICRP Publication 74 dose calculations (neutron & gamma)
- ANSI/ANS-6.4.3 shielding with Geometric Progression buildup factors
- NIST XCOM attenuation coefficients
- Simulation file I/O for Geant4 (.root), FLUKA (.flair), MCNP (.mctal),
  SERPENT (.det), and raw text/CSV formats
- Publication-quality plotting (Nature, thesis, presentation styles)

When answering:
• Be concise and precise.  Cite standards (ICRP 74, ANSI 6.4.3, NIST XCOM)
  when relevant.
• If the user provides data context below, use it to ground your answers.
• Suggest triples-sigfast CLI commands or Python API calls where helpful.
"""

_FILE_CONTEXT_TEMPLATE = """
**Loaded file context:**
- File: {file_path}
- Energy range: {e_min:.3f} – {e_max:.3f} MeV
- Total counts: {total_counts:.2e}
- Peaks detected: {peak_count}
- Peak energies (MeV): {peak_energies}
- Mean relative error (R): {mre:.4f}
- Converged (R < 0.05): {converged}
"""


# ---------------------------------------------------------------------------
# File analysis helper
# ---------------------------------------------------------------------------


def _build_file_context(file_path: str) -> str:
    """Read *file_path* with ``SimReader`` and return a context string."""
    from triples_sigfast.core.signal import find_peaks, savitzky_golay
    from triples_sigfast.io import SimReader
    from triples_sigfast.stats.mc import is_converged, mean_relative_error

    reader = SimReader(file_path)
    counts, energies = reader.get_spectrum()

    smoothed = savitzky_golay(counts, window=11, polyorder=3)
    peaks = find_peaks(
        smoothed,
        min_height=float(smoothed.max()) * 0.05,
        min_distance=10,
    )
    mre = float(mean_relative_error(counts))
    converged = bool(is_converged(counts).all())

    peak_energies = (
        ", ".join(f"{energies[p]:.3f}" for p in peaks[:10])
        if len(peaks) > 0
        else "none"
    )
    if len(peaks) > 10:
        peak_energies += f" ... (+{len(peaks) - 10} more)"

    return _FILE_CONTEXT_TEMPLATE.format(
        file_path=file_path,
        e_min=float(energies[0]),
        e_max=float(energies[-1]),
        total_counts=float(counts.sum()),
        peak_count=len(peaks),
        peak_energies=peak_energies,
        mre=mre,
        converged="YES" if converged else "NO — run more histories",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_chat(file_path: str | None = None) -> None:
    """Launch the interactive sigfast copilot chat.

    Parameters
    ----------
    file_path : str or None
        Optional path to a simulation file.  If provided the file is analysed
        and the results are injected into the LLM system prompt.
    """

    # -- dependency gate ------------------------------------------------
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        console.print(
            "[red]Error:[/red] The [bold]google-genai[/bold] package "
            "is required for the chat copilot.\n"
            "Install it with:\n\n"
            "  [cyan]pip install google-genai[/cyan]\n"
        )
        sys.exit(1)

    # -- API key --------------------------------------------------------
    try:
        from triples_sigfast.cli.config import get_api_key
    except ImportError:
        console.print(
            "[red]Error:[/red] Could not import the config module.  "
            "Make sure triples-sigfast is installed correctly."
        )
        sys.exit(1)

    api_key = get_api_key()
    if not api_key:
        console.print(
            "[red]Error:[/red] No Gemini API key found.\n\n"
            "Set one with:\n"
            "  [cyan]sigfast config set-key YOUR_KEY[/cyan]\n"
            "or export the environment variable:\n"
            "  [cyan]export SIGFAST_API_KEY=YOUR_KEY[/cyan]\n"
        )
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # -- Build system prompt -------------------------------------------
    system_prompt = _SYSTEM_PROMPT_BASE

    if file_path is not None:
        console.print(f"[dim]Analysing [cyan]{file_path}[/cyan] …[/dim]")
        try:
            file_context = _build_file_context(file_path)
            system_prompt += file_context
            console.print("[green]File context loaded into copilot.[/green]\n")
        except Exception as exc:
            console.print(
                f"[yellow]Warning:[/yellow] Could not analyse file: {exc}\n"
                "Continuing without file context.\n"
            )

    # -- Initialise chat session ----------------------------------------
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
    )
    chat_session = client.chats.create(
        model="gemini-2.0-flash",
        config=config,
    )

    # -- Welcome banner -------------------------------------------------
    console.print(
        "[bold cyan]sigfast copilot[/bold cyan] — "
        "Ask questions about your physics data."
    )
    console.print("Type 'exit' or 'quit' to leave.\n")

    # -- REPL loop ------------------------------------------------------
    while True:
        try:
            user_input = console.input("[bold green][sigfast] > [/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            # Attempt streaming for a responsive feel.
            response = chat_session.send_message_stream(stripped)
            full_text = ""
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text
            console.print()
            console.print(Markdown(full_text))
            console.print()
        except Exception:
            # Fall back to non-streaming on any streaming error.
            try:
                response = chat_session.send_message(stripped)
                console.print()
                console.print(Markdown(response.text))
                console.print()
            except Exception as inner_exc:
                console.print(f"\n[red]Error:[/red] {inner_exc}\n")
