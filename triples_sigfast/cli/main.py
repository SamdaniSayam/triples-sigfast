"""
triples_sigfast.cli.main
------------------------
Command-line interface entry point for triples-sigfast.

Each subcommand is implemented as a Click command attached to the root `cli`
group.  All heavy imports (NumPy, Numba-compiled modules, I/O backends) are
deferred to the body of each command function so that `--help` and
`--version` remain instant.

Available subcommands
---------------------
    sigfast analyze  <file>
    sigfast compare  <file1> <file2> ...
    sigfast dose     --flux FLUX --energy ENERGY --particle PARTICLE
    sigfast shield   --material MATERIAL --thickness THICKNESS --energy ENERGY
    sigfast report   <file1> <file2> ...
    sigfast info
    sigfast welcome
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

# Module-level console used by all commands.
console = Console()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _banner():  # pragma: no cover
    """Render the welcome banner without animation (used as a command header)."""
    from .welcome import print_welcome

    print_welcome(animated=False)


# ---------------------------------------------------------------------------
# Root command group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.version_option(package_name="triples-sigfast")
@click.pass_context
def cli(ctx):  # pragma: no cover
    """triples-sigfast -- GIL-free physics simulation analysis engine.

    Run without a subcommand to display the interactive welcome page.
    Run 'sigfast --help' to list all available commands.
    """
    # If no subcommand was provided, show the full animated welcome page.
    if ctx.invoked_subcommand is None:
        from .welcome import print_welcome

        print_welcome(animated=True)


# ---------------------------------------------------------------------------
# sigfast info
# ---------------------------------------------------------------------------


@cli.command()
def info():  # pragma: no cover
    """Show the installed version, available modules, and standards compliance."""
    _banner()

    from .. import __version__

    # Build a table listing every sub-package and its current status.
    module_table = Table(
        title="Installed Modules",
        show_header=True,
        header_style="bold blue",
    )
    module_table.add_column("Module", style="cyan")
    module_table.add_column("Description")
    module_table.add_column("Status", justify="center")

    modules = [
        ("triples_sigfast.core", "JIT-compiled signal processing", "available"),
        ("triples_sigfast.stats", "Monte Carlo statistics", "available"),
        ("triples_sigfast.nuclear", "Nuclear physics (ICRP/ANSI/NIST)", "available"),
        ("triples_sigfast.io", "Simulation file readers", "available"),
        ("triples_sigfast.viz", "Publication-quality plots", "available"),
        ("triples_sigfast.cli", "Command-line interface", "available"),
        ("triples_sigfast.detectors", "Detector physics", "planned v2.0"),
        ("triples_sigfast.plasma", "Plasma physics", "planned v2.0"),
    ]
    for mod, desc, status in modules:
        module_table.add_row(mod, desc, status)

    console.print(f"\n[bold]Version:[/bold] {__version__}")
    console.print(module_table)

    # Build a second table showing the nuclear/physics standards that the
    # library implements.
    standards_table = Table(
        title="Standards Compliance",
        show_header=True,
        header_style="bold green",
    )
    standards_table.add_column("Domain")
    standards_table.add_column("Standard")

    for domain, standard in [
        ("Dose conversion", "ICRP Publication 74 (1996)"),
        ("Buildup factors", "ANSI/ANS-6.4.3-1991"),
        ("Attenuation", "NIST XCOM database"),
        ("Isotope data", "NUBASE2020"),
        ("Convergence", "MCNP standard (R < 0.05)"),
    ]:
        standards_table.add_row(domain, standard)

    console.print(standards_table)


# ---------------------------------------------------------------------------
# sigfast welcome
# ---------------------------------------------------------------------------


@cli.command()
def welcome():  # pragma: no cover
    """Display the triples-sigfast welcome page with feature overview.

    \b
    This page is also shown automatically when running `sigfast` with no
    arguments.

    \b
    Example:
        sigfast welcome
    """
    from .welcome import print_welcome

    print_welcome(animated=True)


# ---------------------------------------------------------------------------
# sigfast analyze
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--key", default=None, help="Histogram or tally key inside the file.")
@click.option(
    "--window",
    default=11,
    show_default=True,
    help="Savitzky-Golay filter window (must be odd).",
)
@click.option(
    "--polyorder", default=3, show_default=True, help="Savitzky-Golay polynomial order."
)
@click.option(
    "--threshold",
    default=0.05,
    show_default=True,
    help="Monte Carlo convergence threshold (mean relative error R).",
)
@click.option(
    "--output", default=None, help="Save spectrum plot to this file (PDF/PNG/SVG)."
)
@click.option(
    "--term-plot/--no-term-plot",
    default=True,
    show_default=True,
    help="Render spectrum plot directly in the terminal.",
)
def analyze(file, key, window, polyorder, threshold, output, term_plot):  # pragma: no cover
    """Analyze a simulation output file or raw data file.

    Reads the file, validates Monte Carlo convergence, smooths the energy
    spectrum using a Savitzky-Golay filter, detects spectral peaks, and
    prints a formatted summary table.

    \b
    Supported simulation formats:
        .root   -- Geant4 (via uproot)
        .flair  -- FLUKA
        .mctal  -- MCNP
        .det    -- SERPENT

    \b
    Supported raw data formats (no code required):
        .csv, .tsv, .txt, .dat, .asc, .out

    \b
    Examples:
        sigfast analyze simulation.root
        sigfast analyze output.mctal --key tally_4
        sigfast analyze spectrum.csv --output figure.pdf
        sigfast analyze data.txt --key col1 --output plot.png
    """
    _banner()

    # Deferred imports keep startup time low and avoid loading heavy
    # backends (uproot, Numba) for commands that do not need them.
    from ..core.signal import find_peaks, savitzky_golay
    from ..io import SimReader
    from ..stats.mc import is_converged, mean_relative_error

    # ---------- Step 1: Read the file ----------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Reading file...", total=None)

        try:
            reader = SimReader(file)
        except Exception as exc:
            console.print(f"[red]Error reading file:[/red] {exc}")
            sys.exit(1)

        # ---------- Step 2: Extract the energy spectrum ----------
        progress.update(task, description="Extracting spectrum...")
        try:
            counts, energies = reader.get_spectrum(key)
        except Exception as exc:
            console.print(f"[red]Error extracting spectrum:[/red] {exc}")
            console.print("Try specifying a column or key with --key. Available keys:")
            reader.summary()
            sys.exit(1)

        # ---------- Step 3: Run the analysis pipeline ----------
        progress.update(task, description="Running analysis...")

        # Compute mean relative error -- the standard MC convergence metric.
        mre = float(mean_relative_error(counts))

        # is_converged() returns a boolean array (one value per bin).
        # All bins must satisfy R < threshold for the run to be considered converged.
        conv = is_converged(counts, threshold=threshold)
        converged = bool(conv.all())

        # Smooth the raw spectrum to reduce statistical noise while preserving
        # peak positions and amplitudes.
        smoothed = savitzky_golay(counts, window=window, polyorder=polyorder)

        # Detect peaks above 5 % of the maximum -- avoids noise spikes being
        # reported as physics peaks.
        peaks = find_peaks(
            smoothed,
            min_height=float(smoothed.max()) * 0.05,
            min_distance=10,
        )

        progress.update(task, description="Done.", completed=True)

    # ---------- Step 4: Display results ----------
    if term_plot:
        import plotext as plt

        plt.clf()
        plt.theme("clear")
        plt.scatter(energies, counts, label="Raw counts", marker="dot", color="blue")
        plt.plot(energies, smoothed, label="Smoothed", color="red")
        if len(peaks) > 0:
            peak_energies = energies[peaks]
            peak_counts = smoothed[peaks]
            plt.scatter(
                peak_energies, peak_counts, marker="x", color="yellow", label="Peaks"
            )
        plt.title(f"Energy Spectrum -- {Path(file).name}")
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.show()
        console.print()

    result_table = Table(
        title=f"Analysis: {file}",
        show_header=True,
        header_style="bold cyan",
    )
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", justify="right")

    result_table.add_row("Total counts", f"{counts.sum():.2e}")
    result_table.add_row("Energy bins", str(len(counts)))
    result_table.add_row("Energy range", f"{energies[0]:.3f} - {energies[-1]:.3f} MeV")
    result_table.add_row("Mean relative error", f"{mre:.4f}")
    result_table.add_row(
        f"Converged (R<{threshold:.2f})",
        "[green]YES[/green]" if converged else "[red]NO -- run more histories[/red]",
    )
    result_table.add_row("Peaks detected", str(len(peaks)))

    # Report up to five peak energies; indicate if more exist.
    if len(peaks) > 0:
        peak_str = ", ".join(f"{energies[p]:.3f}" for p in peaks[:5])
        if len(peaks) > 5:
            peak_str += f" ... (+{len(peaks) - 5} more)"
        result_table.add_row("Peak energies (MeV)", peak_str)

    console.print(result_table)

    # ---------- Step 5: Optionally save the spectrum plot ----------
    if output:
        from ..viz import PhysicsPlot

        plot = PhysicsPlot(style="publication")
        plot.spectrum(
            energies,
            counts,
            smoothed=smoothed,
            peaks=peaks,
            title=f"Energy Spectrum -- {Path(file).name}",
        )
        plot.save(output)
        console.print(f"[green]Plot saved:[/green] {output}")


# ---------------------------------------------------------------------------
# sigfast compare
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--labels",
    default=None,
    help="Comma-separated display labels (must match file count).",
)
@click.option(
    "--energy", default=1.25, show_default=True, help="Reference gamma energy in MeV."
)
@click.option(
    "--output",
    default="comparison.pdf",
    show_default=True,
    help="Output plot filename.",
)
@click.option(
    "--term-plot/--no-term-plot",
    default=True,
    show_default=True,
    help="Render overlay plot directly in the terminal.",
)
def compare(files, labels, energy, output, term_plot):  # pragma: no cover
    """Compare multiple simulation output files on a single overlay plot.

    Applies Savitzky-Golay smoothing to every spectrum, ranks materials by
    lowest spectral standard deviation, and saves an overlay plot.

    \b
    Examples:
        sigfast compare co2.root pb.root fe.root
        sigfast compare co2.root pb.root --labels CO2,Lead --output result.pdf
    """
    _banner()

    if len(files) < 2:
        console.print("[red]Error:[/red] Provide at least 2 files to compare.")
        sys.exit(1)

    # Build the label list; fall back to filename stems if not provided.
    label_list = labels.split(",") if labels else [Path(f).stem for f in files]
    if len(label_list) != len(files):
        console.print("[red]Error:[/red] Number of labels must match number of files.")
        sys.exit(1)

    from ..core.signal import savitzky_golay
    from ..io import SimReader
    from ..stats.mc import mean_relative_error

    # ---------- Read and process each file ----------
    results = []
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        for filepath, label in zip(files, label_list):
            task = progress.add_task(f"Reading {label}...", total=None)

            reader = SimReader(filepath)
            counts, energies = reader.get_spectrum()

            # Smooth with default SG parameters (window=11, order=3).
            smoothed = savitzky_golay(counts, window=11, polyorder=3)

            # Standard deviation of the smoothed spectrum is used as the
            # ranking metric -- lower deviation implies better attenuation.
            deviation = float(np.std(smoothed))
            mre = float(mean_relative_error(counts))

            results.append(
                {
                    "label": label,
                    "file": filepath,
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

    # ---------- Rank and display ----------
    if term_plot:
        import plotext as plt

        plt.clf()
        plt.theme("clear")
        for r in results:
            plt.plot(r["energies"], r["smoothed"], label=r["label"])
        plt.title("Shielding Material Comparison")
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.show()
        console.print()

    # Sort ascending by deviation; rank 1 is the best-performing material.
    sorted_results = sorted(results, key=lambda x: x["deviation"])

    rank_table = Table(
        title="Material Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    rank_table.add_column("Material", style="cyan")
    rank_table.add_column("Total Counts", justify="right")
    rank_table.add_column("Deviation (std)", justify="right")
    rank_table.add_column("Mean R", justify="right")
    rank_table.add_column("Rank", justify="center")

    for rank, r in enumerate(sorted_results, 1):
        marker = "[bold green]BEST[/bold green]" if rank == 1 else str(rank)
        rank_table.add_row(
            r["label"],
            f"{r['counts'].sum():.2e}",
            f"{r['deviation']:.4f}",
            f"{r['mre']:.4f}",
            marker,
        )

    console.print(rank_table)
    console.print(
        f"\n[bold green]Best material:[/bold green] "
        f"{sorted_results[0]['label']} "
        f"(lowest deviation: {sorted_results[0]['deviation']:.4f})"
    )

    # ---------- Generate and save the overlay plot ----------
    from ..viz import PhysicsPlot

    plot = PhysicsPlot(style="publication")
    plot.compare_spectra(
        results,
        title="Shielding Material Comparison",
        xlabel="Energy (MeV)",
        ylabel="Counts",
    )
    plot.save(output)
    console.print(f"[green]Plot saved:[/green] {output}")


# ---------------------------------------------------------------------------
# sigfast dose
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--flux", required=True, type=float, help="Particle flux in particles/cm2/s."
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
    """Calculate ambient dose equivalent rate from particle flux.

    Uses ICRP Publication 74 (1996) fluence-to-dose conversion coefficients
    with log-log interpolation between tabulated energy points.

    \b
    Examples:
        sigfast dose --flux 1e6 --energy 2.35 --particle neutron
        sigfast dose --flux 5e5 --energy 1.25 --particle gamma
    """
    _banner()

    from ..core.signal import flux_to_dose

    # flux_to_dose delegates to nuclear/dose.py for the ICRP 74 data tables.
    dose_rate = flux_to_dose(flux=flux, energy_mev=energy, particle=particle)

    dose_table = Table(
        title="ICRP 74 Dose Calculation",
        show_header=True,
        header_style="bold cyan",
    )
    dose_table.add_column("Parameter", style="cyan")
    dose_table.add_column("Value", justify="right")

    # The ICRP occupational limit is 20 mSv/year averaged over 5 years,
    # which corresponds to approximately 2000 uSv/hr for a 40-hour work week.
    dose_table.add_row("Particle type", particle)
    dose_table.add_row("Energy", f"{energy:.4f} MeV")
    dose_table.add_row("Flux", f"{flux:.3e} particles/cm2/s")
    dose_table.add_row("Dose rate", f"[bold green]{dose_rate:.6f} uSv/hr[/bold green]")
    dose_table.add_row("ICRP occ. limit", "2000 uSv/hr")
    dose_table.add_row(
        "Within limit?",
        "[green]YES[/green]"
        if dose_rate < 2000
        else "[red]EXCEEDS OCCUPATIONAL LIMIT[/red]",
    )

    console.print(dose_table)
    console.print("\n[dim]Standard: ICRP Publication 74, 1996[/dim]")


# ---------------------------------------------------------------------------
# sigfast shield
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--material",
    required=True,
    type=click.Choice(
        ["lead", "iron", "concrete", "water", "polyethylene", "aluminum"]
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
    help="Source geometry for buildup factor selection.",
)
def shield(material, thickness, energy, geometry):  # pragma: no cover
    """Calculate gamma ray transmission through a shielding layer.

    Applies the ANSI/ANS-6.4.3 Geometric Progression buildup factor model,
    which accounts for scattered radiation that Beer-Lambert (simple exponential
    attenuation) does not include.

    \b
    Examples:
        sigfast shield --material lead --thickness 10 --energy 1.25
        sigfast shield --material concrete --thickness 30 --geometry plane_source
    """
    _banner()

    from ..nuclear.shielding import _get_mu, attenuation_with_buildup

    # Transmission with GP buildup correction (includes scattered photons).
    T_buildup = attenuation_with_buildup(thickness, material, energy, geometry)

    # Linear attenuation coefficient from the NIST XCOM database.
    mu = _get_mu(material, energy)

    # Derived quantities used in the results table.
    mfp = mu * thickness  # mean free paths traversed
    T_bl = float(np.exp(-mu * thickness))  # Beer-Lambert (no scatter)
    hvl = np.log(2) / mu  # half-value layer in cm

    shield_table = Table(
        title="ANSI/ANS-6.4.3 Shielding Calculation",
        show_header=True,
        header_style="bold cyan",
    )
    shield_table.add_column("Parameter", style="cyan")
    shield_table.add_column("Value", justify="right")

    shield_table.add_row("Material", material.title())
    shield_table.add_row("Thickness", f"{thickness:.2f} cm")
    shield_table.add_row("Gamma energy", f"{energy:.4f} MeV")
    shield_table.add_row("Geometry", geometry)
    shield_table.add_row("Linear atten. coeff.", f"{mu:.4f} cm-1")
    shield_table.add_row("Mean free paths", f"{mfp:.3f} mfp")
    shield_table.add_row("Half-value layer (HVL)", f"{hvl:.3f} cm")
    shield_table.add_row("Transmission (Beer-Lambert)", f"{T_bl * 100:.4f}%")
    shield_table.add_row(
        "Transmission (GP buildup)",
        f"[bold green]{T_buildup * 100:.4f}%[/bold green]",
    )
    shield_table.add_row("Dose reduction", f"{(1 - T_buildup) * 100:.2f}%")
    shield_table.add_row(
        "Buildup correction",
        f"{T_buildup / T_bl:.3f}x (scattered radiation factor)",
    )

    console.print(shield_table)
    console.print("\n[dim]Standards: ANSI/ANS-6.4.3-1991  |  NIST XCOM[/dim]")


# ---------------------------------------------------------------------------
# sigfast report
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--output",
    default="shielding_report.pdf",
    show_default=True,
    help="Output PDF filename.",
)
@click.option(
    "--title",
    default="triples-sigfast Analysis Report",
    show_default=True,
    help="Report title.",
)
@click.option(
    "--author",
    default="TripleS Studio",
    show_default=True,
    help="Author name for the report header.",
)
def report(files, output, title, author):  # pragma: no cover
    """Generate an automated PDF report for one or more simulation files.

    Runs the full analysis pipeline on each file and writes a formatted PDF
    containing spectrum plots, Monte Carlo convergence tables, and dose
    assessment sections.

    \b
    Examples:
        sigfast report simulation.root --output report.pdf
        sigfast report co2.root pb.root fe.root --title "Shielding Study"
    """
    _banner()

    from .report import AutoReport

    rep = AutoReport(title=title, author=author)

    # Register each file with a label derived from its filename stem.
    for filepath in files:
        label = Path(filepath).stem
        rep.add_simulation(filepath, label=label)
        console.print(f"  Added: [cyan]{label}[/cyan] ({filepath})")

    # Run the full pipeline and write the PDF.
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Generating PDF report...", total=None)
        try:
            rep.generate(output)
            progress.update(task, description="[green]Done.[/green]", completed=True)
        except Exception as exc:
            console.print(f"[red]Error generating report:[/red] {exc}")
            sys.exit(1)

    console.print(f"\n[bold green]Report saved:[/bold green] {output}")


# ---------------------------------------------------------------------------
# sigfast guide
# ---------------------------------------------------------------------------


@cli.command()
def guide():  # pragma: no cover
    """Interactive guided workflow for first-time users.

    Walks through four steps in the terminal: loading a file, analysing the
    spectrum, calculating dose, and saving a publication plot.  No scripting
    is required.

    \b
    Example:
        sigfast guide
    """
    _banner()

    console.print("\n[bold blue]triples-sigfast Interactive Guide[/bold blue]")
    console.print("This will walk you through a complete analysis, step by step.\n")

    # ------------------------------------------------------------------
    # Step 1 -- Load the simulation or data file
    # ------------------------------------------------------------------
    console.print("[bold]Step 1 of 4 -- Load your simulation or data file[/bold]")
    console.print(
        "[dim]Supported: .root / .flair / .mctal / .det / "
        ".csv / .tsv / .txt / .dat / .asc / .out[/dim]"
    )
    file = click.prompt("Enter path to your file", type=click.Path())

    try:
        from ..io import SimReader

        reader = SimReader(file)
        console.print(
            f"[green]File loaded.[/green]  Format detected: {reader.format.upper()}"
        )
        reader.summary()
    except Exception as exc:
        console.print(f"[red]Could not load file:[/red] {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2 -- Extract and analyse the energy spectrum
    # ------------------------------------------------------------------
    console.print("\n[bold]Step 2 of 4 -- Extract and analyse the spectrum[/bold]")

    # The --key option maps to histogram keys (ROOT/MCNP) or column names
    # (CSV/TXT).  An empty string means "use first available".
    key = click.prompt(
        "Histogram or column key (press Enter for auto-detect)", default=""
    )
    key = key if key else None

    from ..core.signal import find_peaks, savitzky_golay
    from ..stats.mc import is_converged, mean_relative_error

    counts, energies = reader.get_spectrum(key)
    smoothed = savitzky_golay(counts, window=11, polyorder=3)
    peaks = find_peaks(
        smoothed, min_height=float(smoothed.max()) * 0.05, min_distance=10
    )
    mre = float(mean_relative_error(counts))
    converged = bool(is_converged(counts).all())

    console.print(f"  Bins:         {len(counts)}")
    console.print(f"  Energy range: {energies[0]:.3f} - {energies[-1]:.3f} MeV")
    console.print(f"  Mean R:       {mre:.4f}")
    console.print(
        f"  Converged:    "
        f"{'[green]YES[/green]' if converged else '[red]NO -- more histories needed[/red]'}"
    )
    console.print(f"  Peaks found:  {len(peaks)}")

    # ------------------------------------------------------------------
    # Step 3 -- Dose calculation
    # ------------------------------------------------------------------
    console.print("\n[bold]Step 3 of 4 -- Dose calculation (ICRP 74)[/bold]")
    energy_mev = click.prompt("Source energy in MeV", default=2.35, type=float)
    particle = click.prompt(
        "Particle type",
        type=click.Choice(["neutron", "gamma"]),
        default="neutron",
    )

    from ..core.signal import flux_to_dose

    dose_rate = flux_to_dose(
        flux=float(counts.sum()),
        energy_mev=energy_mev,
        particle=particle,
    )
    console.print(f"  [bold green]Dose rate: {dose_rate:.4f} uSv/hr[/bold green]")

    # ------------------------------------------------------------------
    # Step 4 -- Save a publication-quality plot
    # ------------------------------------------------------------------
    console.print(
        "\n[bold]Step 4 of 4 -- Save a publication-quality spectrum plot[/bold]"
    )
    style = click.prompt(
        "Plot style",
        type=click.Choice(["publication", "nature", "thesis", "presentation"]),
        default="publication",
    )
    output = click.prompt("Output filename", default="spectrum_analysis.pdf")

    from ..viz import PhysicsPlot

    plot = PhysicsPlot(style=style)
    plot.spectrum(
        energies,
        counts,
        smoothed=smoothed,
        peaks=peaks,
        title="Energy Spectrum Analysis",
    )
    plot.save(output)

    # Final summary
    console.print("\n[bold green]Analysis complete.[/bold green]")
    console.print(f"  Plot saved:  {output}")
    console.print(f"  Dose rate:   {dose_rate:.4f} uSv/hr")
    console.print(f"  Peaks found: {len(peaks)}")
    console.print("\n[dim]triples-sigfast  |  pip install triples-sigfast[/dim]")
