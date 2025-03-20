import numpy as np
import math
import ROOT
from ROOT import RooFit, RooRealVar, RooArgusBG, RooGaussian, RooDataSet, RooArgList, RooCategory, RooSimultaneous, RooAddPdf

import sys
sys.path.append('../include')
import myHeader as myH

ROOT.gSystem.Load("libRooFit")
ROOT.gROOT.SetBatch(True)

# kOrangeC  = ROOT.TColor.GetColor('#ff7f00')
# kBlueC = ROOT.TColor.GetColor('#1f78b4')
kBlueC = ROOT.kBlue+2
kGreenC = ROOT.kSpring+3

def simultaneousFit(df_se, df_mixingDeuteron, df_mixingProton, mcparas, nBins, ptlims, lowMassLim = 2.96, highMassLim = 3.04, simBkgFit = True, title = '', signalPdf = 'dscb', uncorr_bkgPdf = 'pol1', corr_bkgPdf = 'dscb', df_column = 'fMass'):
  
  signalFitOptions = ['gauss', 'dscb']
  if signalPdf not in signalFitOptions:
    raise ValueError(
      f'Invalid signalPdf option. Expected one of: {signalFitOptions}')
  
  uncorrBkgFitOptions = ['pol0', 'pol1', 'pol2', 'expo']
  if uncorr_bkgPdf not in uncorrBkgFitOptions:
    raise ValueError(
      f'Invalid uncorr_bkgPdf option. Expected one of: {uncorrBkgFitOptions}')
  
  corrBkgFitOptions = ['argus', 'dscb', 'atan', 'landau', 'pol3', 'pol4', 'pol5', 'test']
  if corr_bkgPdf not in corrBkgFitOptions:
    raise ValueError(
      f'Invalid corr_bkgPdf option. Expected one of: {corrBkgFitOptions}')
  
  # bin width for axis title
  bin_width = round((highMassLim - lowMassLim)/nBins, 4)
  
  # Define the mass variable
  x = RooRealVar("x", "Invariant Mass", lowMassLim, highMassLim)
  xShifted =  ROOT.RooFormulaVar("xShifted", "xShifted", "x-3.04", ROOT.RooArgList(x))
  xMirrored = ROOT.RooFormulaVar("xMirrored", "xMirrored", "3.08-x", ROOT.RooArgList(x))

  # Convert pandas data to RooFit dataset
  data_se = myH.ndarray2roo(np.array(df_se['fM']), x, 'data_se')
  data_mixingDeuteron = myH.ndarray2roo(np.array(df_mixingDeuteron[df_column]), x, 'data_mixingDeuteron')
  data_mixingProton = myH.ndarray2roo(np.array(df_mixingProton[df_column]), x, 'data_mixingProton')

  # Category to distinguish datasets
  sample = RooCategory("sample", "Sample Type")
  sample.defineType("mixed_deuteron")
  sample.defineType("mixed_proton")
  sample.defineType("same_event")

  # Combine background data sets
  combData = RooDataSet("combData", "Combined Data", RooArgList(x), RooFit.Index(sample), RooFit.Import("mixed_deuteron", data_mixingDeuteron), RooFit.Import("mixed_proton", data_mixingProton), RooFit.Import("same_event", data_se))

  ############### Define Signal function ###############
  if signalPdf == 'gauss':
    mu = RooRealVar("#mu", "Mean of Gaussian", mcparas[0].getVal(), 2.986, 2.996) # 2.991
    sigma = RooRealVar("#sigma", "Width of Gaussian", 0.0005, 0.0030)
    # fix sigma
    sigma.setVal(mcparas[1].getVal())
    sigma.setConstant(ROOT.kTRUE)
    signal = ROOT.RooGaussian("signal", "signal", x, mu, sigma)
  elif signalPdf == 'dscb':
    mu = ROOT.RooRealVar("#mu", "mu", mcparas[0].getVal(), 2.986, 2.996)
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

  ############### Define uncorrelated bkg function ###############
  if uncorr_bkgPdf == 'pol0':
    uncorr_bkg = ROOT.RooPolynomial("uncorr_bkg", "uncorr_bkg", x, ROOT.RooArgList())
  elif uncorr_bkgPdf == 'pol1':
    c1 = ROOT.RooRealVar('c1', 'c1', 3, 0, 15.3)
    uncorr_bkg = ROOT.RooPolynomial("uncorr_bkg", "uncorr_bkg", xShifted, ROOT.RooArgList(c1))
  elif uncorr_bkgPdf == 'pol2':
    pol_a = ROOT.RooRealVar('pol_a', 'pol_a', 3, -30, 30) #0-20
    pol_b = ROOT.RooRealVar('pol_b', 'pol_b', 3, -30, 30) #0-20
    uncorr_bkg = ROOT.RooPolynomial("uncorr_bkg", "uncorr_bkg", xShifted, ROOT.RooArgList(pol_a, pol_b))
  elif uncorr_bkgPdf == 'expo':
    c1 = ROOT.RooRealVar('c1', 'c1', -35., -100., -10.)
    uncorr_bkg = ROOT.RooExponential('uncorr_bkg', 'uncorr_bkg', x, c1)

  ############### Define correlated bkg function ###############
  if corr_bkgPdf == "argus":
    if ptlims[0] == 2.0:
      argus_m0 = ROOT.RooRealVar("argus_m0", "argus_m0", 0.095, 0.07, 0.15)
      argus_c = ROOT.RooRealVar("argus_c", "argus_c", -3, -10, -1) #-10 (2, 2.4 binning)
      argus_p = ROOT.RooRealVar("argus_p", "argus_p", 10, 4.5, 50) # 5
    elif ptlims[0] == 3.0:
      argus_m0 = ROOT.RooRealVar("argus_m0", "argus_m0", 0.095, 0.07, 0.15)
      argus_c = ROOT.RooRealVar("argus_c", "argus_c", -3, -10, -1) #-10 (2, 2.4 binning)
      argus_p = ROOT.RooRealVar("argus_p", "argus_p", 10, 4.5, 50) # 5
    corr_bkg = ROOT.RooArgusBG("corr_bkg", "corr_bkg", xMirrored, argus_m0, argus_c, argus_p)
  elif corr_bkgPdf == "dscb":
    bkg_mu = ROOT.RooRealVar("bkg_mu", "bkg_mu", 2.994, 2.986, 3.01)
    bkg_sigma = ROOT.RooRealVar("bkg_sigma", "bkg_sigma", 0.0005, 0.01)
    bkg_a1 = ROOT.RooRealVar("bkg_a_{1}", "bkg_a1", 0.1, 2)
    bkg_n1 = ROOT.RooRealVar("bkg_n_{1}", "bkg_n1", 10, 0.1, 20)
    bkg_a2 = ROOT.RooRealVar("bkg_a_{2}", "bkg_a2", 0.1, 2)
    bkg_n2 = ROOT.RooRealVar("bkg_n_{2}", "bkg_n2", 0.1, 10)
    corr_bkg = ROOT.RooCrystalBall("corr_bkg", "corr_bkg", x, bkg_mu, bkg_sigma, bkg_a1, bkg_n1, bkg_a2, bkg_n2)
  elif corr_bkgPdf == "atan":
    atan_a = ROOT.RooRealVar("atan_a", "atan_a", 0.3, 0.00001, 1000)
    atan_b = ROOT.RooRealVar("atan_b", "atan_b", -1, -1000, 0)
    corr_bkg = ROOT.RooATan("corr_bkg", "corr_bkg", x, atan_a, atan_b)
  elif corr_bkgPdf == "landau":
    landau_m0 = ROOT.RooRealVar("landau_m0", "landau_m0", 2.996, 2.986, 3.01)
    landau_sigma = ROOT.RooRealVar("landau_sigma", "landau_sigma", 0.0005, 0.1)
    corr_bkg = ROOT.RooLandau("corr_bkg", "corr_bkg", x, landau_m0, landau_sigma)
  elif corr_bkgPdf == "pol5":
    bkg_a1 = ROOT.RooRealVar("bkg_a1", "bkg_a1", -1., 1.)
    bkg_a2 = ROOT.RooRealVar("bkg_a2", "bkg_a2", -1., 1.)
    bkg_a3 = ROOT.RooRealVar("bkg_a3", "bkg_a3", -1., 1.)
    bkg_a4 = ROOT.RooRealVar("bkg_a4", "bkg_a4", -1., 1.)
    bkg_a5 = ROOT.RooRealVar("bkg_a5", "bkg_a5", -1., 1.)
    corr_bkg = ROOT.RooChebychev("corr_bkg", "corr_bkg", x, ROOT.RooArgList(bkg_a1, bkg_a2, bkg_a3, bkg_a4, bkg_a5))
  elif corr_bkgPdf == "pol4":
    bkg_a1 = ROOT.RooRealVar("bkg_a1", "bkg_a1", 0.9, -1., 1.)
    bkg_a2 = ROOT.RooRealVar("bkg_a2", "bkg_a2", -0.3,-1., 1.)
    bkg_a3 = ROOT.RooRealVar("bkg_a3", "bkg_a3", -0.3,-1., 1.)
    bkg_a4 = ROOT.RooRealVar("bkg_a4", "bkg_a4", 0.1, -1., 1.)
    corr_bkg = ROOT.RooChebychev("corr_bkg", "corr_bkg", x, ROOT.RooArgList(bkg_a1, bkg_a2, bkg_a3, bkg_a4))
  elif corr_bkgPdf == "pol3":
    bkg_a1 = ROOT.RooRealVar("bkg_a1", "bkg_a1", 0.9, -1., 1.)
    bkg_a2 = ROOT.RooRealVar("bkg_a2", "bkg_a2", -0.3, -1., 1.)
    bkg_a3 = ROOT.RooRealVar("bkg_a3", "bkg_a3", -0.4, -1., 1.)
    corr_bkg = ROOT.RooChebychev("corr_bkg", "corr_bkg", x, ROOT.RooArgList(bkg_a1, bkg_a2, bkg_a3))
  # elif corr_bkgPdf == "test":
  #   bkg1_mu = ROOT.RooRealVar("bkg1_mu", "bkg1_mu", 2.999, 2.997, 3.001)
  #   bkg1_sigma = ROOT.RooRealVar("bkg1_sigma", "bkg1_sigma", 0.0005, 0.01)
  #   bkg1_a1 = ROOT.RooRealVar("bkg1_a_{1}", "bkg1_a1", 0.1, 2)
  #   bkg1_n1 = ROOT.RooRealVar("bkg1_n_{1}", "bkg1_n1", 10, 0.1, 20)
  #   bkg1_a2 = ROOT.RooRealVar("bkg1_a_{2}", "bkg1_a2", 0.1, 2)
  #   bkg1_n2 = ROOT.RooRealVar("bkg1_n_{2}", "bkg1_n2", 0.1, 10)
  #   bkg2_mu = ROOT.RooRealVar("bkg2_mu", "bkg2_mu", 2.994, 2.986, 3.01)
  #   bkg2_sigma = ROOT.RooRealVar("bkg2_sigma", "bkg2_sigma", 0.0005, 0.01)
  #   bkg2_a1 = ROOT.RooRealVar("bkg2_a_{1}", "bkg2_a1", 0.1, 2)
  #   bkg2_n1 = ROOT.RooRealVar("bkg2_n_{1}", "bkg2_n1", 10, 0.1, 20)
  #   bkg2_a2 = ROOT.RooRealVar("bkg2_a_{2}", "bkg2_a2", 0.1, 2)
  #   bkg2_n2 = ROOT.RooRealVar("bkg2_n_{2}", "bkg2_n2", 0.1, 10)
  #   corr_bkg1 = ROOT.RooCrystalBall("corr_bkg1", "corr_bkg1", x, bkg1_mu, bkg1_sigma, bkg1_a1, bkg1_n1, bkg1_a2, bkg1_n2)
  #   corr_bkg2 = ROOT.RooCrystalBall("corr_bkg2", "corr_bkg2", x, bkg2_mu, bkg2_sigma, bkg2_a1, bkg2_n1, bkg2_a2, bkg2_n2)
  #   frac_corr_peak1 = RooRealVar("frac_corr_peak1", "Correlated bkg fraction peak 1", 0.5, 0.0, 1.0)
  #   corr_bkg = ROOT.RooAddPdf("corr_bkg", "corr_bkg", RooArgList(corr_bkg1, corr_bkg2), RooArgList(frac_corr_peak1))

  frac_corr_mixedDeuteron = RooRealVar("frac_corr_mixedDeuteron", "Correlated background fraction while mixing deuterons", 0.01, 0.0, 1.0)
  if simBkgFit == True:
    # Define composite model for mixingDeuteron background
    total_bkg_pdf = RooAddPdf("total_bkg_pdf", "Uncorr. + Corr. Background", RooArgList(corr_bkg, uncorr_bkg), RooArgList(frac_corr_mixedDeuteron))
  elif simBkgFit == False:
    # Define correlated bkg only model for mixingDeuteron background
    total_bkg_pdf = corr_bkg


  # Define composite model for SE background
  frac_corr = RooRealVar("frac_corr", "Correlated background fraction", 0.01, 0.0, 1.0)
  total_sig_bkg_pdf = RooAddPdf("total_sig_bkg_pdf", "Uncorr. + Corr. Background", RooArgList(uncorr_bkg, corr_bkg), RooArgList(frac_corr))

  # Define composite model for SE total
  frac_sig = RooRealVar("frac_sig", "Signal peak fraction", 0.5, 0.0, 1.0)
  total_pdf = RooAddPdf("total_pdf", "Signal + Uncorr. + Corr. Background", RooArgList(signal, total_sig_bkg_pdf), RooArgList(frac_sig))

  # Simultaneous PDF setup
  simPdf = RooSimultaneous("simPdf", "Simultaneous PDF", sample)
  simPdf.addPdf(uncorr_bkg, "mixed_proton")  # Background only fit to mixingProton
  simPdf.addPdf(total_bkg_pdf, "mixed_deuteron")  # Background fit to mixingDeuteron
  simPdf.addPdf(total_pdf, "same_event")  # Signal + Background fit to SE

  # Perform simultaneous fit
  simPdf.fitTo(combData)

  ############################################
  ############### Draw results ###############
  ############################################

  ############### Draw uncorrelated background spectrum with fit ###############
  canvas_bkg_uncorr = ROOT.TCanvas("canvas_bkg_uncorr" + title, "Background uncorrelated", 800, 800)
  frame_bkg_uncorr = x.frame(RooFit.Bins(nBins))  # Apply binning
  data_mixingProton.plotOn(frame_bkg_uncorr, RooFit.Binning(nBins))
  uncorr_bkg.plotOn(frame_bkg_uncorr, RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(kOrangeC))
  frame_bkg_uncorr.GetYaxis().SetTitle(f'Counts / ({bin_width}' + ' GeV/#it{c}^{2})')
  frame_bkg_uncorr.GetXaxis().SetTitle('#it{M}(p+#pi+d) (GeV/#it{c}^{2})')
  frame_bkg_uncorr.SetTitle(canvas_bkg_uncorr.GetTitle() + " " + title)  # Set title for upper plot
  label = ROOT.TPaveText(0.15, 0.25, 0.25, 0.35, "NDC")
  chi2Val = frame_bkg_uncorr.chiSquare()
  label.AddText(f"#chi^{{2}} = {chi2Val:.2f}")
  label.SetFillColor(0)
  label.SetBorderSize(0)
  frame_bkg_uncorr.addObject(label)
  canvas_bkg_uncorr.cd()
  frame_bkg_uncorr.Draw()

  ############### Draw correlated background spectrum with fit ###############
  canvas_bkg_corr = ROOT.TCanvas("canvas_bkg_corr" + title, "Background correlated", 800, 800)
  frame_bkg_corr = x.frame(RooFit.Bins(nBins))  # Apply binning
  data_mixingDeuteron.plotOn(frame_bkg_corr, RooFit.Binning(nBins))
  total_bkg_pdf.plotOn(frame_bkg_corr, RooFit.LineColor(kBlueC))
  total_bkg_pdf.paramOn(frame_bkg_corr, Layout = [0.9, 0.6, 0.9])
  if simBkgFit == True:
    total_bkg_pdf.plotOn(frame_bkg_corr, RooFit.Components("uncorr_bkg"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(kOrangeC))
  # total_bkg_pdf.paramOn(frame_bkg_corr)
  frame_bkg_corr.GetYaxis().SetTitle(f'Counts / ({bin_width}' + ' GeV/#it{c}^{2})')
  frame_bkg_corr.GetXaxis().SetTitle('#it{M}(p+#pi+d) (GeV/#it{c}^{2})')
  frame_bkg_corr.SetTitle(canvas_bkg_corr.GetTitle() + " " + title)  # Set title for upper plot
  label = ROOT.TPaveText(0.15, 0.25, 0.25, 0.35, "NDC")
  chi2Val = frame_bkg_corr.chiSquare()
  label.AddText(f"#chi^{{2}} = {chi2Val:.2f}")
  label.SetFillColor(0)
  label.SetBorderSize(0)
  frame_bkg_corr.addObject(label)
  canvas_bkg_corr.cd()
  frame_bkg_corr.Draw()

  ############### Draw total fit ###############
  canvas_signal = ROOT.TCanvas("canvas_signal" + title, "SE Fit Results", 800, 1000)
  pad1 = ROOT.TPad("pad1", "Fit", 0, 0.3, 1, 1)  # Increase pad1 size
  pad2 = ROOT.TPad("pad2", "Residual", 0, 0, 1, 0.3)  # Decrease pad2 size
  pad1.SetBottomMargin(0.02)  # Reduce space between pads
  pad2.SetTopMargin(0)  # Remove top margin of lower pad
  pad2.SetBottomMargin(0.3)  # Increase space for x-axis
  pad1.Draw()
  pad2.Draw()

  # get number of signal and background
  signal_region_low = mu.getVal() - 3 * sigma.getVal()
  signal_region_high = mu.getVal() + 3 * sigma.getVal()
  x.setRange("signal", signal_region_low, signal_region_high)
  nSignal = frac_sig.getVal() * data_se.sumEntries()
  nSignal_err = nSignal * frac_sig.getError() / frac_sig.getVal()
  varBkg = total_bkg_pdf.createIntegral(ROOT.RooArgSet(x),ROOT.RooFit.NormSet(x),ROOT.RooFit.Range("signal"))
  expNBkg = data_se.sumEntries() * (1 - frac_sig.getVal()) * varBkg.getVal()
  unc_bkg = data_se.sumEntries() * varBkg.getVal() * frac_sig.getError()
  hRawYield = ROOT.TH1F("hRawYield", "hRawYield", 1, 0, 1)
  hRawYield.SetBinContent(1, nSignal)
  hRawYield.SetBinError(1, nSignal_err)

  # Avoid NaN
  if math.isnan(expNBkg):
    expNBkg = 0
  if math.isnan(unc_bkg):
    unc_bkg = 0
  # Get S/B
  if (expNBkg > 0):
    signal_to_bkg = nSignal / expNBkg
    signal_to_bkg_err = math.sqrt( math.pow(nSignal_err / expNBkg, 2) + math.pow( unc_bkg * nSignal / (expNBkg * expNBkg), 2) )
    signal_to_bkg_str = str(round(signal_to_bkg, 2)) + ' #pm ' + str(round(signal_to_bkg_err, 2))
  else:
    signal_to_bkg_str = 'NaN'
  
  # Get significance
  if (nSignal + expNBkg > 0):
    significance = nSignal / (math.sqrt(nSignal + expNBkg))
    significance_err = math.sqrt( math.pow( nSignal_err * (nSignal + 2*expNBkg) / (2 * (nSignal + expNBkg) * math.sqrt(nSignal + expNBkg)), 2) + math.pow( unc_bkg * nSignal / (2 * math.pow(nSignal + expNBkg, 1.5)), 2) )
    significance_str = str(round(significance, 2)) + ' #pm ' + str(round(significance_err, 2))
  else:
    significance_str = 'NaN'

  # Upper plot (full fit)
  pad1.cd()
  frame1 = x.frame(RooFit.Bins(nBins))  # Apply binning
  data_se.plotOn(frame1, RooFit.Binning(nBins), RooFit.Name("data"))  # Set binning for data
  total_pdf.plotOn(frame1, RooFit.Components("uncorr_bkg"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(kGreenC))
  total_pdf.plotOn(frame1, RooFit.Components("total_sig_bkg_pdf"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(kOrangeC), RooFit.Name("background"))
  total_pdf.plotOn(frame1, RooFit.LineColor(kBlueC))
  frame1.SetTitle(canvas_signal.GetTitle() + " " + title)  # Set title for upper plot
  frame1.GetXaxis().SetTitle("")  # Remove x-axis title in upper plot
  frame1.GetXaxis().SetLabelOffset(999)
  frame1.GetXaxis().SetLabelSize(0)
  frame1.GetYaxis().SetTitle(f'Counts / ({bin_width}' + ' GeV/c^{2})')
  frame1.GetYaxis().SetTitleSize(0.05)
  frame1.GetYaxis().SetLabelSize(0.04)
  frame1.GetYaxis().SetTitleOffset(1.2)
  frame1.Draw()
  # Info
  paveText = ROOT.TPaveText(0.14, 0.3, 0.7, 0.86, "NDC")
  paveText.SetName("paveText")
  paveText.SetBorderSize(0)
  paveText.SetFillStyle(0)
  paveText.SetTextFont(42)
  paveText.SetTextAlign(11)
  # paveText.SetTextSize(28)
  paveText.AddText('LHC24am, LHC24an, LHC24ao pass1')
  paveText.AddText('{0}'.format(ptlims[0]) + ' < #it{p}_{T} < ' + '{0}'.format(ptlims[1]) + ' GeV/#it{c}')
  paveText.AddText('')
  paveText.AddText('#mu = ' + str(round(mu.getValV(), 5)) + ' #pm ' + str(round(mu.getError(), 5)))
  paveText.AddText('#sigma = ' + str(round(sigma.getValV(), 5)) + ' #pm ' + str(round(sigma.getError(), 5)))
  paveText.AddText('')
  chi2Val = frame1.chiSquare()
  paveText.AddText(f"#chi^{{2}} = {chi2Val:.2f}")
  paveText.AddText('S = ' + str(round(nSignal)) + ' #pm ' + str(round(nSignal_err)))
  paveText.AddText('B(3#sigma) = ' + str(round(expNBkg)) + ' #pm ' + str(round(unc_bkg)))
  paveText.AddText('S/B(3#sigma) = ' + signal_to_bkg_str)
  paveText.AddText('Significance(3#sigma) = ' + significance_str)
  frame1.addObject(paveText)
  paveText.Draw()

  pad2.cd()
  residuals = frame1.residHist("data", "background")
  frame_residuals = x.frame(RooFit.Bins(nBins))
  frame_residuals.addPlotable(residuals, "PE")  # Add residuals properly
  signal.plotOn(frame_residuals, RooFit.LineColor(ROOT.kRed), RooFit.LineStyle(ROOT.kDashed), RooFit.Name("signal"))
  frame_residuals.SetTitle("")
  frame_residuals.GetXaxis().SetTitle('#it{M}(p+#pi+d) (GeV/#it{c}^{2})')
  frame_residuals.GetXaxis().SetTitleSize(0.1)
  frame_residuals.GetXaxis().SetLabelSize(0.1)
  frame_residuals.GetXaxis().SetTitleOffset(0.9)
  frame_residuals.GetYaxis().SetTitle("Residuals")
  frame_residuals.GetYaxis().SetTitleSize(0.1)
  frame_residuals.GetYaxis().SetLabelSize(0.08)
  frame_residuals.GetYaxis().SetTitleOffset(0.6)
  frame_residuals.SetMarkerStyle(20)
  frame_residuals.SetMarkerSize(0.8)
  frame_residuals.SetLineColor(ROOT.kBlack)
  frame_residuals.Draw()

  # Add dashed gray line at y=0
  line = ROOT.TLine(2.96, 0, 3.02, 0)
  line.SetLineColor(ROOT.kGray+2)
  line.SetLineStyle(2)
  line.SetLineWidth(2)
  pad2.cd()
  line.Draw()

  # Update canvas
  canvas_signal.Update()
  canvas_signal.RedrawAxis()

  return (hRawYield, canvas_bkg_uncorr, canvas_bkg_corr, canvas_signal, nSignal, nSignal_err, expNBkg)