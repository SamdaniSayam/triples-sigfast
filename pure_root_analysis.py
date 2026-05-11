import glob
import os
import sys

import ROOT


def main():
    # Run in batch mode to prevent X11 windows from opening
    ROOT.gROOT.SetBatch(True)

    # Setup canvas
    canvas = ROOT.TCanvas("c1", "Stacked Shielding Comparison", 1200, 800)
    canvas.SetLogy()
    canvas.SetGrid()

    # Setup legend
    legend = ROOT.TLegend(0.55, 0.65, 0.88, 0.88)
    legend.SetHeader("Shielding Materials", "C")

    # File paths
    file_pattern = "/home/megalith/Research/Nuclear_Shielding/data/geant4_output/*.root"
    files = glob.glob(file_pattern)

    if not files:
        print("No files found!")
        sys.exit(1)

    histograms = []
    spectrum_analyzer = ROOT.TSpectrum(10)

    # Standard ROOT color palette
    colors = [
        ROOT.kRed,
        ROOT.kBlue,
        ROOT.kGreen + 2,
        ROOT.kMagenta,
        ROOT.kOrange + 7,
        ROOT.kCyan + 1,
    ]

    for i, filepath in enumerate(files):
        # 1. Open File
        f = ROOT.TFile(filepath, "READ")

        # 2. Extract Histogram
        hist = f.Get("NeutronEnergy")

        # ROOT memory management: Detach from file so it survives closing
        hist.SetDirectory(0)
        f.Close()

        # 3. Process: ROOT's built-in Smoothing (Markov Chain method)
        hist.Smooth(1)

        # 4. Process: ROOT's TSpectrum Peak Detection
        num_peaks = spectrum_analyzer.Search(hist, 2, "goff", 0.05)

        # 5. Styling
        color = colors[i % len(colors)]
        hist.SetLineColor(color)
        hist.SetLineWidth(2)

        # Extract material name from file path
        label = os.path.basename(filepath).replace(".root", "").replace("_output", "")
        hist.SetTitle("Pure CERN ROOT Analysis;Energy (MeV);Counts")

        # Need to set max/min manually for overlapping histograms in ROOT
        if i == 0:
            hist.SetMaximum(1e4)
            hist.SetMinimum(1)
            hist.Draw("HIST")
        else:
            hist.Draw("HIST SAME")

        # Draw peaks
        if num_peaks > 0:
            x_peaks = spectrum_analyzer.GetPositionX()
            y_peaks = spectrum_analyzer.GetPositionY()
            # ROOT requires creating TPolyMarker for custom peak drawing over HIST
            pm = ROOT.TPolyMarker(num_peaks, x_peaks, y_peaks)
            pm.SetMarkerStyle(23)  # Triangle down
            pm.SetMarkerColor(color)
            pm.SetMarkerSize(1.5)
            pm.Draw("SAME")
            # Keep reference alive
            histograms.append(pm)

        legend.AddEntry(hist, label, "l")
        histograms.append(hist)

    legend.Draw()

    # Save the output
    output_path = "/home/megalith/sigfast/pure_root_stacked.png"
    canvas.SaveAs(output_path)
    print(f"CERN ROOT analysis complete. Graph saved to {output_path}")


if __name__ == "__main__":
    main()
