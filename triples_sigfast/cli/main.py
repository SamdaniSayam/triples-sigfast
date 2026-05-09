"""
triples_sigfast.cli.main
─────────────────────────
Command-line interface for triples-sigfast.

Usage:
    sigfast analyze  <file>
    sigfast compare  <file1> <file2> ...
    sigfast dose     --flux FLUX --energy ENERGY --particle PARTICLE
    sigfast shield   --material MATERIAL --thickness THICKNESS --energy ENERGY
    sigfast report   <file1> <file2> ...
    sigfast info
    sigfast guide
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def _banner():  # pragma: no cover
    from triples_sigfast.cli.welcome import print_welcome

    print_welcome(animated=False)


# ── Root command group ────────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.version_option(package_name="triples-sigfast")
@click.pass_context
def cli(ctx):  # pragma: no cover
    """triples-sigfast — GIL-free physics simulation analysis engine.

    Run without a subcommand to see the welcome page.
    Run 'sigfast --help' to list all commands.
    """
    if ctx.invoked_subcommand is None:
        from triples_sigfast.cli.welcome import print_welcome

        print_welcome(animated=True)


# ── sigfast info ──────────────────────────────────────────────────────────────


@cli.command()
def info():  # pragma: no cover
    """Show library info, installed version, and available modules."""
    _banner()

    import triples_sigfast as sf

    table = Table(
        title="Installed Modules",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Module", style="cyan")
    table.add_column("Description")
    table.add_column("Status", justify="center")

    modules = [
        ("triples_sigfast.core", "JIT-compiled signal processing", "✅"),
        ("triples_sigfast.stats", "Monte Carlo statistics", "✅"),
        ("triples_sigfast.nuclear", "Nuclear physics (ICRP/ANSI/NIST)", "✅"),
        ("triples_sigfast.io", "Simulation file readers", "✅"),
        ("triples_sigfast.viz", "Publication-quality plots", "✅"),
        ("triples_sigfast.cli", "Command-line interface", "✅"),
        ("triples_sigfast.detectors", "Detector physics", "🔄 v2.0"),
        ("triples_sigfast.plasma", "Plasma physics", "🔄 v2.0"),
    ]
    for mod, desc, status in modules:
        table.add_row(mod, desc, status)

    console.print(f"\n[bold]Version:[/bold] {sf.__version__}")
    console.print(table)

    standards = Table(
        title="Standards Compliance",
        show_header=True,
        header_style="bold green",
    )
    standards.add_column("Module")
    standards.add_column("Standard")
    for mod, std in [
        ("Dose conversion", "ICRP Publication 74 (1996)"),
        ("Buildup factors", "ANSI/ANS-6.4.3-1991"),
        ("Attenuation", "NIST XCOM database"),
        ("Isotope data", "NUBASE2020"),
        ("Convergence", "MCNP standard (R < 0.05)"),
    ]:
        standards.add_row(mod, std)
    console.print(standards)


# ── sigfast welcome ───────────────────────────────────────────────────────────


@cli.command()
def welcome():  # pragma: no cover
    """
    Show the triples-sigfast welcome page.

    Displays the full ASCII logo, feature overview, performance metrics,
    quick-start commands, and a random physics quote.

    \b
    This is also shown automatically when you run `sigfast` with no arguments.

    \b
    Example:
        sigfast welcome
    """
    from triples_sigfast.cli.welcome import print_welcome

    print_welcome(animated=True)


# ── sigfast analyze ───────────────────────────────────────────────────────────


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--key", default=None, help="Histogram key inside file.")
@click.option("--window", default=11, show_default=True, help="SG filter window.")
@click.option("--polyorder", default=3, show_default=True, help="SG polynomial order.")
@click.option(
    "--threshold", default=0.05, show_default=True, help="MC convergence threshold."
)
@click.option("--output", default=None, help="Save plot to file (PDF/PNG/SVG).")
def analyze(file, key, window, polyorder, threshold, output):  # pragma: no cover
    """
    Analyze a simulation output file.

    Reads the file, checks Monte Carlo convergence, smooths the
    energy spectrum, detects peaks, and displays a summary.

    Supports: .root (Geant4), .flair (FLUKA), .mctal (MCNP), .det (SERPENT)

    \b
    Examples:
        sigfast analyze simulation.root
        sigfast analyze output.mctal --key tally_4
        sigfast analyze neutrons.root --output figure.pdf
    """
    _banner()

    from triples_sigfast import find_peaks, savitzky_golay
    from triples_sigfast.io import SimReader
    from triples_sigfast.stats.mc import (
        is_converged,
        mean_relative_error,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Reading simulation file...", total=None)
        try:
            reader = SimReader(file)
        except Exception as e:
            console.print(f"[red]Error reading file:[/red] {e}")
            sys.exit(1)

        progress.update(task, description="Extracting spectrum...")
        try:
            counts, energies = reader.get_spectrum(key)
        except Exception as e:
            console.print(f"[red]Error extracting spectrum:[/red] {e}")
            console.print("Try specifying a key with --key. Available keys:")
            reader.summary()
            sys.exit(1)

        progress.update(task, description="Running analysis...")
        # R = relative_error(counts)
        mre = float(mean_relative_error(counts))
        conv = is_converged(counts, threshold=threshold)
        converged = bool(conv.all())
        smoothed = savitzky_golay(counts, window=window, polyorder=polyorder)
        peaks = find_peaks(
            smoothed,
            min_height=float(smoothed.max()) * 0.05,
            min_distance=10,
        )
        progress.update(task, description="Done.", completed=True)

    table = Table(
        title=f"Analysis: {file}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total counts", f"{counts.sum():.2e}")
    table.add_row("Energy bins", str(len(counts)))
    table.add_row("Energy range", f"{energies[0]:.3f} – {energies[-1]:.3f} MeV")
    table.add_row("Mean relative error", f"{mre:.4f}")
    table.add_row(
        f"Converged (R<{threshold:.2f})",
        "[green]YES[/green]" if converged else "[red]NO — run more histories[/red]",
    )
    table.add_row("Peaks detected", str(len(peaks)))

    if len(peaks) > 0:
        peak_str = ", ".join(f"{energies[p]:.3f}" for p in peaks[:5])
        if len(peaks) > 5:
            peak_str += f" ... (+{len(peaks) - 5} more)"
        table.add_row("Peak energies (MeV)", peak_str)

    console.print(table)

    if output:
        from triples_sigfast.viz import PhysicsPlot

        plot = PhysicsPlot(style="publication")
        plot.spectrum(
            energies,
            counts,
            smoothed=smoothed,
            peaks=peaks,
            title=f"Energy Spectrum — {file}",
        )
        plot.save(output)
        console.print(f"[green]Plot saved:[/green] {output}")


# ── sigfast compare ───────────────────────────────────────────────────────────


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--labels", default=None, help="Comma-separated labels.")
@click.option("--energy", default=1.25, show_default=True, help="Gamma energy in MeV.")
@click.option(
    "--output", default="comparison.pdf", show_default=True, help="Output plot."
)
def compare(files, labels, energy, output):  # pragma: no cover
    """
    Compare multiple simulation output files.

    Overlays smoothed spectra and ranks by lowest deviation.

    \b
    Examples:
        sigfast compare co2.root pb.root fe.root
        sigfast compare co2.root pb.root --labels CO2,Lead --output result.pdf
    """
    _banner()

    if len(files) < 2:
        console.print("[red]Error:[/red] Provide at least 2 files to compare.")
        sys.exit(1)

    label_list = labels.split(",") if labels else [Path(f).stem for f in files]
    if len(label_list) != len(files):
        console.print("[red]Error:[/red] Number of labels must match number of files.")
        sys.exit(1)

    from triples_sigfast import savitzky_golay
    from triples_sigfast.io import SimReader
    from triples_sigfast.stats.mc import mean_relative_error

    results = []
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        for f, label in zip(files, label_list):
            task = progress.add_task(f"Reading {label}...", total=None)
            reader = SimReader(f)
            counts, energies = reader.get_spectrum()
            smoothed = savitzky_golay(counts, window=11, polyorder=3)
            deviation = float(np.std(smoothed))
            mre = float(mean_relative_error(counts))
            results.append(
                {
                    "label": label,
                    "file": f,
                    "counts": counts,
                    "energies": energies,
                    "smoothed": smoothed,
                    "deviation": deviation,
                    "mre": mre,
                }
            )
            progress.update(
                task, completed=True, description=f"[green]{label} done[/green]"
            )

    table = Table(
        title="Material Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Material", style="cyan")
    table.add_column("Total Counts", justify="right")
    table.add_column("Deviation (std)", justify="right")
    table.add_column("Mean R", justify="right")
    table.add_column("Rank", justify="center")

    sorted_results = sorted(results, key=lambda x: x["deviation"])
    for rank, r in enumerate(sorted_results, 1):
        marker = "[bold green]BEST[/bold green]" if rank == 1 else str(rank)
        table.add_row(
            r["label"],
            f"{r['counts'].sum():.2e}",
            f"{r['deviation']:.4f}",
            f"{r['mre']:.4f}",
            marker,
        )
    console.print(table)
    console.print(
        f"\n[bold green]Best material:[/bold green] "
        f"{sorted_results[0]['label']} "
        f"(lowest deviation: {sorted_results[0]['deviation']:.4f})"
    )

    from triples_sigfast.viz import PhysicsPlot

    plot = PhysicsPlot(style="publication")
    for r in results:
        plot.spectrum(
            r["energies"],
            r["counts"],
            smoothed=r["smoothed"],
            title="Material Comparison",
            label_counts=r["label"] + " (raw)",
            label_smoothed=r["label"] + " (smoothed)",
        )
    plot.save(output)
    console.print(f"[green]Plot saved:[/green] {output}")


# ── sigfast dose ──────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--flux", required=True, type=float, help="Particle flux in particles/cm²/s."
)
@click.option("--energy", required=True, type=float, help="Particle energy in MeV.")
@click.option(
    "--particle",
    default="neutron",
    show_default=True,
    type=click.Choice(["neutron", "gamma"]),
    help="Particle type.",
)
def dose(flux, energy, particle):  # pragma: no cover
    """
    Calculate biological dose from particle flux.

    Uses ICRP Publication 74 (1996) conversion coefficients.

    \b
    Examples:
        sigfast dose --flux 1e6 --energy 2.35 --particle neutron
        sigfast dose --flux 5e5 --energy 1.25 --particle gamma
    """
    _banner()

    from triples_sigfast import flux_to_dose

    result = flux_to_dose(flux=flux, energy_mev=energy, particle=particle)

    table = Table(
        title="ICRP 74 Dose Calculation",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Particle type", particle)
    table.add_row("Energy", f"{energy:.4f} MeV")
    table.add_row("Flux", f"{flux:.3e} particles/cm²/s")
    table.add_row("Dose rate", f"[bold green]{result:.6f} μSv/hr[/bold green]")
    table.add_row("ICRP occ. limit", "2000 μSv/hr")
    table.add_row(
        "Safe?",
        "[green]YES[/green]"
        if result < 2000
        else "[red]EXCEEDS OCCUPATIONAL LIMIT[/red]",
    )

    console.print(table)
    console.print("\n[dim]Standard: ICRP Publication 74, 1996[/dim]")


# ── sigfast shield ────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--material",
    required=True,
    type=click.Choice(
        [
            "lead",
            "iron",
            "concrete",
            "water",
            "polyethylene",
            "aluminum",
        ]
    ),
    help="Shielding material.",
)
@click.option("--thickness", required=True, type=float, help="Shield thickness in cm.")
@click.option(
    "--energy", default=1.25, show_default=True, type=float, help="Gamma energy in MeV."
)
@click.option(
    "--geometry",
    default="point_source",
    show_default=True,
    type=click.Choice(["point_source", "plane_source", "infinite_slab"]),
    help="Source geometry.",
)
def shield(material, thickness, energy, geometry):  # pragma: no cover
    """
    Calculate gamma ray transmission through a shield.

    Uses ANSI/ANS-6.4.3 Geometric Progression buildup factors.

    \b
    Examples:
        sigfast shield --material lead --thickness 10 --energy 1.25
        sigfast shield --material concrete --thickness 30 --geometry plane_source
    """
    _banner()

    from triples_sigfast.nuclear.shielding import _get_mu, attenuation_with_buildup

    T = attenuation_with_buildup(thickness, material, energy, geometry)
    mu = _get_mu(material, energy)
    mfp = mu * thickness
    T_bl = float(np.exp(-mu * thickness))
    hvl = np.log(2) / mu

    table = Table(
        title="ANSI/ANS-6.4.3 Shielding Calculation",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Material", material.title())
    table.add_row("Thickness", f"{thickness:.2f} cm")
    table.add_row("Gamma energy", f"{energy:.4f} MeV")
    table.add_row("Geometry", geometry)
    table.add_row("Linear atten. coeff.", f"{mu:.4f} cm⁻¹")
    table.add_row("Mean free paths", f"{mfp:.3f} mfp")
    table.add_row("HVL", f"{hvl:.3f} cm")
    table.add_row("Transmission (Beer-Lambert)", f"{T_bl * 100:.4f}%")
    table.add_row(
        "Transmission (GP buildup)", f"[bold green]{T * 100:.4f}%[/bold green]"
    )
    table.add_row("Dose reduction", f"{(1 - T) * 100:.2f}%")
    table.add_row("Buildup correction", f"{T / T_bl:.3f}× (scattered radiation)")

    console.print(table)
    console.print("\n[dim]Standard: ANSI/ANS-6.4.3-1991 · NIST XCOM[/dim]")


# ── sigfast report ────────────────────────────────────────────────────────────


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--output",
    default="shielding_report.pdf",
    show_default=True,
    help="Output PDF report filename.",
)
@click.option(
    "--title",
    default="triples-sigfast Analysis Report",
    show_default=True,
    help="Report title.",
)
@click.option(
    "--author", default="TripleS Studio", show_default=True, help="Author name."
)
def report(files, output, title, author):  # pragma: no cover
    """
    Generate an automated PDF report for simulation files.

    Runs the full analysis pipeline and writes a formatted PDF with
    spectrum plots, MC convergence tables, and dose assessment.

    \b
    Examples:
        sigfast report simulation.root --output report.pdf
        sigfast report co2.root pb.root fe.root --title "Shielding Study"
    """
    _banner()

    from triples_sigfast.cli.report import AutoReport

    rep = AutoReport(title=title, author=author)
    for f in files:
        label = Path(f).stem
        rep.add_simulation(f, label=label)
        console.print(f"  Added: [cyan]{label}[/cyan] ({f})")

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Generating PDF report...", total=None)
        try:
            rep.generate(output)
            progress.update(task, description="[green]Done.[/green]", completed=True)
        except Exception as e:
            console.print(f"[red]Error generating report:[/red] {e}")
            sys.exit(1)

    console.print(f"\n[bold green]Report saved:[/bold green] {output}")


# ── sigfast guide ─────────────────────────────────────────────────────────────


@cli.command()
def guide():  # pragma: no cover
    """
    Interactive guided workflow for beginners.

    Walks you through loading a simulation file, analyzing the
    spectrum, calculating dose, and saving a publication plot.

    \b
    Example:
        sigfast guide
    """
    _banner()

    console.print(
        "\n[bold blue]Welcome to the triples-sigfast guided workflow![/bold blue]"
    )
    console.print("This will walk you through a complete analysis step by step.\n")

    # Step 1 — Load file
    console.print("[bold]Step 1 of 4 — Load your simulation file[/bold]")
    file = click.prompt(
        "Enter path to your simulation file (.root/.flair/.mctal/.det)",
        type=click.Path(),
    )
    try:
        from triples_sigfast.io import SimReader

        reader = SimReader(file)
        console.print(f"[green]File loaded.[/green] Format: {reader.format.upper()}")
        reader.summary()
    except Exception as e:
        console.print(f"[red]Could not load file:[/red] {e}")
        sys.exit(1)

    # Step 2 — Extract spectrum
    console.print("\n[bold]Step 2 of 4 — Extract and analyse spectrum[/bold]")
    key = click.prompt("Histogram key (press Enter for first found)", default="")
    key = key if key else None

    from triples_sigfast import find_peaks, savitzky_golay
    from triples_sigfast.stats.mc import is_converged, mean_relative_error

    counts, energies = reader.get_spectrum(key)
    smoothed = savitzky_golay(counts, window=11, polyorder=3)
    peaks = find_peaks(
        smoothed, min_height=float(smoothed.max()) * 0.05, min_distance=10
    )
    mre = float(mean_relative_error(counts))
    converged = bool(is_converged(counts).all())

    console.print(f"  Bins:         {len(counts)}")
    console.print(f"  Energy range: {energies[0]:.3f} – {energies[-1]:.3f} MeV")
    console.print(f"  Mean R:       {mre:.4f}")
    console.print(
        f"  Converged:    "
        f"{'[green]YES[/green]' if converged else '[red]NO — more histories needed[/red]'}"
    )
    console.print(f"  Peaks found:  {len(peaks)}")

    # Step 3 — Dose
    console.print("\n[bold]Step 3 of 4 — Dose calculation[/bold]")
    energy_mev = click.prompt("Source energy in MeV", default=2.35, type=float)
    particle = click.prompt(
        "Particle type",
        type=click.Choice(["neutron", "gamma"]),
        default="neutron",
    )

    from triples_sigfast import flux_to_dose

    dose_rate = flux_to_dose(
        flux=float(counts.sum()),
        energy_mev=energy_mev,
        particle=particle,
    )
    console.print(f"  [bold green]Dose rate: {dose_rate:.4f} μSv/hr[/bold green]")

    # Step 4 — Save plot
    console.print("\n[bold]Step 4 of 4 — Save publication plot[/bold]")
    style = click.prompt(
        "Plot style",
        type=click.Choice(["publication", "nature", "thesis", "presentation"]),
        default="publication",
    )
    output = click.prompt("Output filename", default="spectrum_analysis.pdf")

    from triples_sigfast.viz import PhysicsPlot

    plot = PhysicsPlot(style=style)
    plot.spectrum(
        energies,
        counts,
        smoothed=smoothed,
        peaks=peaks,
        title="Energy Spectrum Analysis",
    )
    plot.save(output)

    console.print("\n[bold green]Analysis complete![/bold green]")
    console.print(f"  Plot saved: {output}")
    console.print(f"  Dose rate:  {dose_rate:.4f} μSv/hr")
    console.print(f"  Peaks:      {len(peaks)} detected")
    console.print("\n[dim]triples-sigfast · pip install triples-sigfast[/dim]")
