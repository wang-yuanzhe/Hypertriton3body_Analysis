import numpy as np
import math
import ROOT
from ROOT import RooFit, RooRealVar, RooArgusBG, RooGaussian, RooDataSet, RooArgList, RooCategory, RooSimultaneous, RooAddPdf

import sys
sys.path.append('../include')
import myHeader as myH

def simultaneousFit(df_se, df_mixing_deuteron, df_mixing_proton, outfile, mcparas = "", title = "", corr_bkgPdf = "DSCB", uncorr_bkgPdf = "pol1"):
    ''' Simultaneous fit to signal and background modeled by mixing events'''

    ############### Initialize the dataset ###############
    # Define the mass variable
    x = RooRealVar("x", "Invariant Mass", 2.96, 3.04)
    xMirrored = ROOT.RooFormulaVar("xMirrored", "xMirrored", "3.08-x", ROOT.RooArgList(x)) # for argus
    xShifted =  ROOT.RooFormulaVar("xShifted", "xShifted", "x-3.04", ROOT.RooArgList(x))   # to define the parameters of pol better

    # Convert pandas data to RooFit dataset
    data_se = myH.ndarray2roo(np.array(df_se['fM']), x, "data_se")
    data_mixing_deuteron = myH.ndarray2roo(np.array(df_mixing_deuteron['fM']), x, "data_mixing_deuteron")
    data_mixing_proton = myH.ndarray2roo(np.array(df_mixing_proton['fM']), x, "data_mixing_proton")

    # Category to distinguish datasets
    sample = RooCategory("sample", "Sample Type")
    sample.defineType("same_event")
    sample.defineType("mixed_deuteron")
    sample.defineType("mixed_proton")

    # Combine datasets
    combData = RooDataSet("combData", "Combined Data", RooArgList(x), RooFit.Index(sample), RooFit.Import("same_event", data_se), RooFit.Import("mixed_deuteron", data_mixing_deuteron), RooFit.Import("mixed_proton", data_mixing_proton))

    ############### Define Signal function ###############
    mu = ROOT.RooRealVar("#mu", "mu", 2.991, 2.986, 2.996)
    sigma = ROOT.RooRealVar("#sigma", "sigma", 0.0005, 0.0030)
    a1 = ROOT.RooRealVar("a_{1}", "a1", 0.1, 2)
    n1 = ROOT.RooRealVar("n_{1}", "n1", 0.1, 10)
    a2 = ROOT.RooRealVar("a_{2}", "a2", 0.1, 2)
    n2 = ROOT.RooRealVar("n_{2}", "n2", 0.1, 10)
    # fix the parameters
    sigma.setVal(mcparas[1].getVal())
    a1.setVal(mcparas[2].getVal())
    n1.setVal(mcparas[3].getVal())
    a2.setVal(mcparas[4].getVal())
    n2.setVal(mcparas[5].getVal())
    sigma.setConstant(ROOT.kTRUE)
    a1.setConstant(ROOT.kTRUE)
    n1.setConstant(ROOT.kTRUE)
    a2.setConstant(ROOT.kTRUE)
    n2.setConstant(ROOT.kTRUE)
    signal = ROOT.RooCrystalBall("signal", "signal", x, mu, sigma, a1, n1, a2, n2)

    ############### Define uncorrected background function (background from mixing proton) ###############
    if uncorr_bkgPdf == "exp":
        c1 = ROOT.RooRealVar('c1', 'c1', 35., 10., 100.)
        uncorr_bkg = ROOT.RooExponential('uncorr_bkg', 'uncorr_bkg', x, c1)
    elif uncorr_bkgPdf == "pol1":
        c1 = ROOT.RooRealVar('c1', 'c1', 3, 0, 12)
        uncorr_bkg = ROOT.RooPolynomial("uncorr_bkg", "uncorr_bkg", xShifted, ROOT.RooArgList(c1))

    ############### Define Mixing Background function ###############
    if corr_bkgPdf == "argus":
        argus_m0 = ROOT.RooRealVar("argus_m0", "argus_m0", 0.095, 0.07, 0.15)
        argus_c = ROOT.RooRealVar("argus_c", "argus_c", -6, -10, -1)
        argus_p = ROOT.RooRealVar("argus_p", "argus_p", 2, 0.5, 10)
        argus_bkg = ROOT.RooArgusBG("argus_bkg", "argus_bkg", xMirrored, argus_m0, argus_c, argus_p)
        corr_bkg = argus_bkg
    elif corr_bkgPdf == "atan":
        atan_a = ROOT.RooRealVar("atan_a", "atan_a", 0.3, 0.00001, 1000)
        atan_b = ROOT.RooRealVar("atan_b", "atan_b", -1, -1000, 0)
        atan_bkg = ROOT.RooATan("atan_bkg", "atan_bkg", x, atan_a, atan_b)
        corr_bkg = atan_bkg
    elif corr_bkgPdf == "Landau":
        landau_m0 = ROOT.RooRealVar("landau_m0", "landau_m0", 2.996, 2.986, 3.01)
        landau_sigma = ROOT.RooRealVar("landau_sigma", "landau_sigma", 0.0005, 0.1)
        landau_bkg = ROOT.RooLandau("landau_bkg", "landau_bkg", x, landau_m0, landau_sigma)
        corr_bkg = landau_bkg
    elif corr_bkgPdf == "DSCB":
        bkg_mu = ROOT.RooRealVar("bkg_#mu", "bkg_mu", 2.996, 2.986, 3.01)
        bkg_sigma = ROOT.RooRealVar("bkg_#sigma", "bkg_sigma", 0.0005, 0.01)
        bkg_a1 = ROOT.RooRealVar("bkg_a_{1}", "bkg_a1", 0.1, 2)
        bkg_n1 = ROOT.RooRealVar("bkg_n_{1}", "bkg_n1", 0.1, 10)
        bkg_a2 = ROOT.RooRealVar("bkg_a_{2}", "bkg_a2", 0.1, 2)
        bkg_n2 = ROOT.RooRealVar("bkg_n_{2}", "bkg_n2", 0.1, 10)
        dscb_bkg = ROOT.RooCrystalBall("dscb_bkg", "dscb_bkg", x, bkg_mu, bkg_sigma, bkg_a1, bkg_n1, bkg_a2, bkg_n2)
        corr_bkg = dscb_bkg
    elif corr_bkgPdf == "pol5": # FIXME: adjust the range of parameters to make it work
        bkg_a1 = ROOT.RooRealVar("bkg_a1", "bkg_a1", -20, 20)
        bkg_a2 = ROOT.RooRealVar("bkg_a2", "bkg_a2", -20, 20)
        bkg_a3 = ROOT.RooRealVar("bkg_a3", "bkg_a3", -20, 20)
        bkg_a4 = ROOT.RooRealVar("bkg_a4", "bkg_a4", -20, 20)
        bkg_a5 = ROOT.RooRealVar("bkg_a5", "bkg_a5", -20, 20)
        poly_bkg = ROOT.RooPolynomial("poly_bkg", "poly_bkg", x, ROOT.RooArgList(bkg_a1, bkg_a2, bkg_a3, bkg_a4, bkg_a5))
        corr_bkg = poly_bkg
    
    frac_corr_mixing_deuteron = RooRealVar("frac_corr_mixing_deuteron", "correlated background fraction while mixing deuterons", 0.5, 0.0, 1.0)
    bkg_mixing_deuteron = ROOT.RooAddPdf("bkg_mixing_deuteron", "bkg_mixing_deuteron", RooArgList(corr_bkg, uncorr_bkg), RooArgList(frac_corr_mixing_deuteron))

    ############### Define Background function ###############
    frac_corr = RooRealVar("frac_corr", "correlated background fraction", 0.5, 0., 1.)
    bkg_pdf = RooAddPdf("bkg_pdf", "Background", RooArgList(corr_bkg, uncorr_bkg), RooArgList(frac_corr))

    ############### Define Composite model  ###############
    fsig = RooRealVar("fsig", "signal peak fraction", 0.5, 0., 1.)
    total_pdf = RooAddPdf("total_pdf", "Signal + Background", RooArgList(signal, bkg_pdf), RooArgList(fsig))

    # Simultaneous PDF setup
    simPdf = RooSimultaneous("simPdf", "Simultaneous PDF", sample)
    simPdf.addPdf(bkg_mixing_deuteron, "mixed_deuteron")  # Background only fit to AO2D_mixing
    simPdf.addPdf(uncorr_bkg, "mixed_proton")  # Extra background fit to AO2D_mixing
    simPdf.addPdf(total_pdf, "same_event")  # Signal + Background fit to AO2D_se

    # Perform simultaneous fit
    simPdf.fitTo(combData)

    ############### Plot the results ###############
    c = ROOT.TCanvas("SE_" + title, "SE", 800, 1000)
    c.cd()
    nBins = 40  # Set the desired number of bins

    frame1 = x.frame(RooFit.Bins(nBins))  # Apply binning
    data_se.plotOn(frame1, RooFit.Binning(nBins))  # Set binning for data
    total_pdf.plotOn(frame1, RooFit.LineColor(ROOT.kBlue))
    chi2Val = frame1.chiSquare()
    total_pdf.plotOn(frame1, RooFit.Components(corr_bkg.GetName()), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    total_pdf.plotOn(frame1, RooFit.Components(uncorr_bkg.GetName()), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))
    # total_pdf.plotOn(frame1, RooFit.Components("signal"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    total_pdf.paramOn(frame1, Layout = [0.2, 0.4, 0.9])

    signal_region_low = mu.getVal() - 3 * sigma.getVal()
    signal_region_high = mu.getVal() + 3 * sigma.getVal()
    x.setRange("signal", signal_region_low, signal_region_high)
    nSignal = fsig.getVal() * data_se.sumEntries()
    nSignal_err = nSignal * fsig.getError() / fsig.getVal()
    varBkg = bkg_pdf.createIntegral(ROOT.RooArgSet(x),ROOT.RooFit.NormSet(x),ROOT.RooFit.Range("signal"))
    expNBkg = data_se.sumEntries() * (1 - fsig.getVal()) * varBkg.getVal()
    unc_bkg = data_se.sumEntries() * fsig.getVal() * varBkg.getVal() * fsig.getError() / fsig.getVal()

    paveText = ROOT.TPaveText(0.65, 0.6, 0.9, 0.9, "NDC")
    paveText.SetName("paveText")
    paveText.SetBorderSize(0)
    paveText.SetFillStyle(0)
    paveText.SetTextFont(4)
    paveText.SetTextAlign(11)
    paveText.SetTextSize(22)
    paveText.AddText('S = ' + str(round(nSignal)) + ' #pm ' + str(round(nSignal_err)))
    paveText.AddText('B(3#sigma) = ' + str(round(expNBkg)) + ' #pm ' + str(round(unc_bkg)))
    if (expNBkg > 0):
        unc_significance = math.sqrt( math.pow(nSignal_err / expNBkg, 2) + math.pow( unc_bkg * nSignal / (expNBkg * expNBkg), 2) )
        paveText.AddText('S/B(3#sigma) = ' + str( round(nSignal / expNBkg, 2) ) + ' #pm ' + str(round(unc_significance, 2)))
    else:
        paveText.AddText('S/B(3#sigma) = NaN') 
    if (nSignal + expNBkg > 0):
        unc_significance = math.sqrt( math.pow( nSignal_err * (nSignal + 2*expNBkg) / (2 * (nSignal + expNBkg) * math.sqrt(nSignal + expNBkg)), 2) + math.pow( unc_bkg * nSignal / (2 * math.pow(nSignal + expNBkg, 1.5)), 2) )
        paveText.AddText('Significance(3#sigma) = ' + str( round(nSignal/(math.sqrt(nSignal + expNBkg)) ,1) ) + ' #pm ' + str(round(unc_significance, 1)) )
    else:
        paveText.AddText('Significance(3#sigma) = NaN')
    frame1.addObject(paveText)
    label = ROOT.TPaveText(0.15, 0.25, 0.25, 0.35, "NDC")
    label.AddText(f"#chi^{{2}} = {chi2Val:.2f}")
    label.SetFillColor(0)
    label.SetBorderSize(0)
    frame1.addObject(label)
    frame1.SetTitle(c.GetTitle() + " " + title)
    frame1.GetXaxis().SetTitle( 'm(p+#pi+d) (GeV/c^{2})' )
    frame1.GetYaxis().SetTitle(frame1.GetYaxis().GetTitle()[:-1] + 'GeV/c^{2} )')
    frame1.Draw()
    paveText.Draw()

    c_mixing_deuteron = ROOT.TCanvas("ME_d " + title, "ME_d", 800, 800)
    c_mixing_deuteron.cd()
    frame3 = x.frame(RooFit.Bins(nBins))  # Apply binning
    data_mixing_deuteron.plotOn(frame3, RooFit.Binning(nBins))  # Set binning for data
    bkg_mixing_deuteron.plotOn(frame3, RooFit.Components(corr_bkg.GetName()), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    bkg_mixing_deuteron.paramOn(frame3, Layout = [0.2, 0.4, 0.9])
    chi2Val = frame3.chiSquare()
    bkg_mixing_deuteron.plotOn(frame3, RooFit.Components(uncorr_bkg.GetName()), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))
    label = ROOT.TPaveText(0.15, 0.25, 0.25, 0.35, "NDC")
    label.AddText(f"#chi^{{2}} = {chi2Val:.2f}")
    label.SetFillColor(0)
    label.SetBorderSize(0)
    frame3.addObject(label)
    frame3.GetXaxis().SetTitle( 'm(p+#pi+d) (GeV/c^{2})' )
    frame3.GetYaxis().SetTitle(frame3.GetYaxis().GetTitle()[:-1] + 'GeV/c^{2} )')
    frame3.SetTitle(c_mixing_deuteron.GetTitle() + " " + title)
    frame3.Draw()

    c_mixing_proton = ROOT.TCanvas("ME_p " + title, "ME_p", 800, 800)
    c_mixing_proton.cd()
    frame4 = x.frame(RooFit.Bins(nBins))  # Apply binning
    data_mixing_proton.plotOn(frame4, RooFit.Binning(nBins))  # Set binning for data
    uncorr_bkg.plotOn(frame4, RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))
    uncorr_bkg.paramOn(frame4, Layout = [0.2, 0.4, 0.9])
    chi2Val = frame4.chiSquare()
    label = ROOT.TPaveText(0.15, 0.25, 0.25, 0.35, "NDC")
    label.AddText(f"#chi^{{2}} = {chi2Val:.2f}")
    label.SetFillColor(0)
    label.SetBorderSize(0)
    frame4.addObject(label)
    frame4.GetXaxis().SetTitle( 'm(p+#pi+d) (GeV/c^{2})' )
    frame4.GetYaxis().SetTitle(frame4.GetYaxis().GetTitle()[:-1] + 'GeV/c^{2} )')
    frame4.SetTitle(c_mixing_proton.GetTitle() + " " + title)
    frame4.Draw()

    outfile.cd()
    c.Write()
    c_mixing_deuteron.Write()
    c_mixing_proton.Write()

    if expNBkg < 0.1: # avoid divide by 0
        expNBkg = 0.1

    return (nSignal, nSignal_err, expNBkg)