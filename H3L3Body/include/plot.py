import ROOT
import numpy as np
import utils

def plot_bdtefficiency_vs_model_output(outdir, modelOutput_x, BDTefficiency_y, pt_bin):
    binkey = f"pT{pt_bin[0]}_{pt_bin[1]}"
    binInfo = f"pT{pt_bin[0]}-{pt_bin[1]} GeV/c"

    c_BDTefficiency = utils.TCanvas(f"c_BDTefficiency{binkey}", "", 800, 600)
    c_BDTefficiency.cd()

    h_Back_BDTeff = ROOT.TH2F(
        f"h_Back_BDTeff{binkey}",
        f"{binInfo};Model_cut;BDT Efficiency",
        1, -10, 15,
        1, 0, 1.1 * np.max(BDTefficiency_y)
    )

    gr_BDTefficiency = ROOT.TGraph(len(modelOutput_x), modelOutput_x, BDTefficiency_y)
    gr_BDTefficiency.SetLineColor(ROOT.kBlue + 2)
    gr_BDTefficiency.SetLineWidth(2)
    gr_BDTefficiency.SetMarkerStyle(20)
    gr_BDTefficiency.SetMarkerColor(ROOT.kBlue + 2)

    h_Back_BDTeff.Draw()
    gr_BDTefficiency.Draw("LP SAME")

    outdir.WriteObject(c_BDTefficiency, f"BDTefficiency{binkey}")
    del c_BDTefficiency, h_Back_BDTeff, gr_BDTefficiency

def plot_significance_times_bdtefficiency(outdir, modelOutput_x, significance_times_BDTEff, binkey, binInfo
):
    c_ModelSelection = utils.TCanvas()
    c_ModelSelection.cd()

    h_Back_modelsel = ROOT.TH2F(
        f"h_Back_modelsel{binkey}",
        f"{binInfo};Model_cut;Expected Significance #times BDT Efficiency",
        1, -10, 15, 1, 0, 1.1 * np.max(significance_times_BDTEff)
    )

    gr_ModelSelection = ROOT.TGraph(
        len(modelOutput_x), modelOutput_x, significance_times_BDTEff
    )

    h_Back_modelsel.Draw()
    gr_ModelSelection.SetLineColor(ROOT.kBlue)
    gr_ModelSelection.Draw("L same")

    outdir.WriteObject(c_ModelSelection, f"ModelSelection_{binkey}")

def plot_signal_to_bdtefficiency(outdir, model_eff_x, SigToBDTefficiency_y, SigToBDTefficiency_errory, maxindex, binkey, binInfo):
    # 3. Prepare TGraphErrors
    gr_SigToBDTefficiency = ROOT.TGraphErrors(
        len(model_eff_x),
        np.array(model_eff_x),
        SigToBDTefficiency_y,
        np.zeros(len(model_eff_x)),
        SigToBDTefficiency_errory
    )
    gr_SigToBDTefficiency1 = ROOT.TGraphErrors(
        len(model_eff_x),
        np.array(model_eff_x),
        SigToBDTefficiency_y
    )

    # 4. Draw canvas
    c_SigToBDTefficiency = utils.TCanvas()
    c_SigToBDTefficiency.cd()

    tf1_sigtoBDTeff = ROOT.TF1("tf1_sigtoBDTeff", "[0]", max(model_eff_x[0], 0.01 * maxindex), min(0.01 * maxindex + 0.2, model_eff_x[-1]))
    tf1_sigtoBDTeff.SetParName(0, "mean")
    tf1_sigtoBDTeff.SetLineColor(ROOT.kRed)
    tf1_sigtoBDTeff.SetLineStyle(2)
    tf1_sigtoBDTeff.SetLineWidth(2)

    gr_SigToBDTefficiency.SetTitle(binInfo + ';Eff_{BDT};N_{Sig} / Eff_{BDT}')
    gr_SigToBDTefficiency.SetFillColor(ROOT.kCyan - 10)
    gr_SigToBDTefficiency.Draw("3A")

    gr_SigToBDTefficiency1.SetLineColor(4)
    gr_SigToBDTefficiency1.SetMarkerStyle(8)
    gr_SigToBDTefficiency1.SetMarkerSize(0.4)
    # gr_SigToBDTefficiency1.Fit("tf1_sigtoBDTeff", "R")
    gr_SigToBDTefficiency1.Draw("same LP")

    outdir.WriteObject(c_SigToBDTefficiency, f"SigToBDTefficiency_{binkey}")
    