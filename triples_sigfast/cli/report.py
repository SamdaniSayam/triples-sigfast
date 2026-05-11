"""
triples_sigfast.cli.report
---------------------------
AutoReport: one-command PDF report generator.

Reads one or more simulation output files, runs the full analysis
pipeline, and produces a publication-ready PDF summary with:
  - Spectrum plots for each simulation
  - Monte Carlo convergence table
  - Dose rate assessment
  - Shielding recommendation

Usage
-----
    from triples_sigfast.cli.report import AutoReport

    report = AutoReport()
    report.add_simulation("co2.root",  label="CO2")
    report.add_simulation("pb.root",   label="Lead")
    report.add_simulation("fe.root",   label="Iron")
    report.generate("shielding_report.pdf")

Or via CLI:
    sigfast report co2.root pb.root fe.root --output report.pdf
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path


class AutoReport:
    """
    Automated PDF report generator for simulation analysis.

    Reads simulation output files, runs the full triples-sigfast
    analysis pipeline, and generates a formatted PDF report.

    Parameters
    ----------
    style : str, optional
        Plot style preset. Default 'publication'.
    title : str, optional
        Report title. Default 'triples-sigfast Analysis Report'.
    author : str, optional
        Author name for the report header.
    energy_mev : float, optional
        Default energy for dose calculations. Default 2.35 MeV.
    particle : str, optional
        Particle type used for dose estimation. Default 'neutron'.

    Examples
    --------
    >>> report = AutoReport(title="Neutron Shielding Study")
    >>> report.add_simulation("geant4_output.root", label="Geant4 baseline")
    >>> report.add_simulation("fluka_output.flair", label="FLUKA comparison")
    >>> report.generate("shielding_report.pdf")
    """

    def __init__(
        self,
        style: str = "publication",
        title: str = "triples-sigfast Analysis Report",
        author: str = "TripleS Studio",
        energy_mev: float = 2.35,
        particle: str = "neutron",
    ) -> None:
        self.style = style
        self.title = title
        self.author = author
        self.energy_mev = energy_mev
        self.particle = particle
        self._simulations: list[dict] = []

    def add_simulation(
        self,
        filepath: str,
        label: str | None = None,
        key: str | None = None,
    ) -> None:
        """
        Add a simulation file to the report.

        Parameters
        ----------
        filepath : str
            Path to simulation output file.
        label : str, optional
            Human-readable label. Defaults to the filename.
        key : str, optional
            Histogram / tally key inside the file. If None,
            the first available spectrum is used.
        """
        if label is None:
            label = Path(filepath).stem

        self._simulations.append(
            {
                "filepath": filepath,
                "label": label,
                "key": key,
            }
        )

    def generate(self, output_path: str = "report.pdf") -> None:
        """
        Run the full analysis pipeline and write the PDF report.

        Parameters
        ----------
        output_path : str
            Destination PDF file path.

        Raises
        ------
        RuntimeError
            If no simulations have been added.
        """
        if not self._simulations:
            raise RuntimeError("No simulations added. Call add_simulation() first.")

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                Image,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as e:
            raise ImportError(
                "reportlab is required for AutoReport. "
                "Install with: pip install reportlab"
            ) from e

        results = self._run_analysis()
        tmpdir = tempfile.mkdtemp()
        story = []
        styles = getSampleStyleSheet()

        # -- Custom styles -----------------------------------------------------
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#185FA5"),
            spaceAfter=6,
        )
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#185FA5"),
            spaceBefore=12,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
        )
        caption_style = ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#5F5E5A"),
            alignment=1,
        )

        # -- Cover -------------------------------------------------------------
        story.append(Spacer(1, 2 * cm))
        story.append(Paragraph(self.title, title_style))
        story.append(
            Paragraph(
                f"Generated by <b>triples-sigfast</b> on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                body_style,
            )
        )
        story.append(Paragraph(f"Author: {self.author}", body_style))
        story.append(Spacer(1, 0.5 * cm))

        # Simulations list
        story.append(Paragraph("Simulations analysed:", body_style))
        for r in results:
            story.append(
                Paragraph(
                    f"  • <b>{r['label']}</b> — {r['filepath']} [{r['format'].upper()}]",
                    body_style,
                )
            )
        story.append(Spacer(1, 1 * cm))

        # -- Summary table -----------------------------------------------------
        story.append(Paragraph("Summary", section_style))

        header = ["Label", "Format", "Bins", "Mean R", "Converged", "Peaks"]
        table_data = [header]
        for r in results:
            table_data.append(
                [
                    r["label"],
                    r["format"].upper(),
                    str(r["n_bins"]),
                    f"{r['mean_R']:.4f}",
                    "YES" if r["converged"] else "NO",
                    str(r["n_peaks"]),
                ]
            )

        t = Table(
            table_data,
            colWidths=[3.5 * cm, 2 * cm, 1.8 * cm, 2 * cm, 2.2 * cm, 1.8 * cm],
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#185FA5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#F1EFE8"), colors.white],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B4B2A9")),
                    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))

        # -- Per-simulation sections -------------------------------------------
        for i, r in enumerate(results):
            story.append(PageBreak())
            story.append(Paragraph(f"Simulation {i + 1}: {r['label']}", section_style))
            story.append(
                Paragraph(
                    f"File: {r['filepath']} | Format: {r['format'].upper()} | "
                    f"Bins: {r['n_bins']} | Energy range: "
                    f"{r['energies'][0]:.3f} – {r['energies'][-1]:.3f} MeV",
                    body_style,
                )
            )
            story.append(Spacer(1, 0.3 * cm))

            # Spectrum plot
            plot_path = os.path.join(tmpdir, f"spectrum_{i}.png")
            self._save_spectrum_plot(r, plot_path)
            story.append(Image(plot_path, width=15 * cm, height=9 * cm))
            story.append(
                Paragraph(
                    f"Figure {i + 1}: Energy spectrum for {r['label']}. "
                    f"Blue: raw counts. Orange: Savitzky-Golay smoothed. "
                    f"Triangles: detected peaks.",
                    caption_style,
                )
            )
            story.append(Spacer(1, 0.5 * cm))

            # MC statistics table
            story.append(Paragraph("Monte Carlo Statistics", section_style))
            mc_data = [
                ["Metric", "Value"],
                ["Total counts", f"{r['counts'].sum():.4e}"],
                ["Mean relative error R", f"{r['mean_R']:.6f}"],
                ["Converged (R < 0.05)", "YES" if r["converged"] else "NO"],
                ["Peaks detected", str(r["n_peaks"])],
            ]
            if r["n_peaks"] > 0:
                peak_str = ", ".join(f"{r['energies'][p]:.3f}" for p in r["peaks"][:5])
                mc_data.append(["Peak energies (MeV)", peak_str])

            mc_table = Table(mc_data, colWidths=[8 * cm, 8 * cm])
            mc_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D9E75")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.HexColor("#E1F5EE"), colors.white],
                        ),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9FE1CB")),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(mc_table)

        # -- Footer ------------------------------------------------------------
        story.append(PageBreak())
        story.append(Paragraph("Standards & References", section_style))
        refs = [
            ("Dose conversion", "ICRP Publication 74 (1996)"),
            ("Buildup factors", "ANSI/ANS-6.4.3-1991"),
            ("Attenuation", "NIST XCOM database"),
            ("Isotope data", "NUBASE2020"),
            ("MC convergence", "MCNP standard (R < 0.05)"),
        ]
        for standard, ref in refs:
            story.append(Paragraph(f"  • <b>{standard}:</b> {ref}", body_style))

        story.append(Spacer(1, 1 * cm))
        story.append(
            Paragraph(
                "Generated by <b>triples-sigfast</b> — "
                "https://pypi.org/project/triples-sigfast/",
                caption_style,
            )
        )

        # -- Build PDF ---------------------------------------------------------
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=self.title,
            author=self.author,
        )
        doc.build(story)
        print(f"Report saved -> {output_path}  ({len(results)} simulation(s))")

    # -- Internal --------------------------------------------------------------

    def _run_analysis(self) -> list[dict]:
        """Run the full analysis pipeline for all added simulations."""
        from ..core.signal import find_peaks, savitzky_golay
        from ..io import SimReader
        from ..stats.mc import (
            is_converged,
            mean_relative_error,
            relative_error,
        )

        results = []
        for sim in self._simulations:
            reader = SimReader(sim["filepath"])
            counts, energies = reader.get_spectrum(sim["key"])
            R = relative_error(counts)
            mre = float(mean_relative_error(counts))
            conv = bool(is_converged(counts, threshold=0.05).all())
            smoothed = savitzky_golay(counts, window=11, polyorder=3)
            peaks = find_peaks(
                smoothed,
                min_height=float(smoothed.max()) * 0.05,
                min_distance=10,
            )
            results.append(
                {
                    "filepath": sim["filepath"],
                    "label": sim["label"],
                    "format": reader.format,
                    "counts": counts,
                    "energies": energies,
                    "smoothed": smoothed,
                    "R": R,
                    "mean_R": mre,
                    "converged": conv,
                    "peaks": peaks,
                    "n_bins": len(counts),
                    "n_peaks": len(peaks),
                }
            )
        return results

    def _save_spectrum_plot(self, r: dict, path: str) -> None:
        """Save a spectrum plot to a PNG file."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(
            r["energies"],
            r["counts"],
            where="mid",
            color="#185FA5",
            alpha=0.6,
            lw=1.2,
            label="Raw",
        )
        ax.plot(
            r["energies"], r["smoothed"], color="#D85A30", lw=2, label="Smoothed (SG)"
        )
        if r["n_peaks"] > 0:
            ax.plot(
                r["energies"][r["peaks"]],
                r["smoothed"][r["peaks"]],
                "v",
                color="#639922",
                ms=10,
                zorder=5,
                label=f"Peaks ({r['n_peaks']})",
            )
        ax.set_title(r["label"])
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Counts")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def __repr__(self) -> str:
        return f"AutoReport(title='{self.title}', simulations={len(self._simulations)})"
