import numpy as np
import math
import ROOT
from ROOT import RooFit, RooRealVar, RooDataSet, RooArgList, RooCategory, RooSimultaneous, RooAddPdf

import utils as utils
import config as cfg

ROOT.gSystem.Load("libRooFit")
ROOT.gROOT.SetBatch(True) # Suppress graphics popups for batch mode

utils.set_style()

def simultaneousFit(df_se, df_mixingDeuteron, df_mixingProton, mcparas, nBins, ptlims, lowMassLim = 2.96, highMassLim = 3.02, simBkgFit = True, title = '', signalPdf = 'dscb', uncorr_bkgPdf = 'pol1', corr_bkgPdf = 'dscb', df_column = 'fMass', corr_bkg_peak = None, fix_bkg_peak = True):
  
  signalFitOptions = ['gauss', 'dscb']
  if signalPdf not in signalFitOptions:
    raise ValueError(
      f'Invalid signalPdf option. Expected one of: {signalFitOptions}')
  
  uncorrBkgFitOptions = ['pol0', 'pol1', 'pol2', 'expo']
  if uncorr_bkgPdf not in uncorrBkgFitOptions:
    raise ValueError(
      f'Invalid uncorr_bkgPdf option. Expected one of: {uncorrBkgFitOptions}')
  
  corrBkgFitOptions = ['argus', 'dscb', 'atan', 'landau', 'pol3', 'pol4', 'pol5']
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
  data_se = utils.ndarray2roo(np.array(df_se['fM']), x, 'data_se')
  data_mixingDeuteron = utils.ndarray2roo(np.array(df_mixingDeuteron[df_column]), x, 'data_mixingDeuteron')
  data_mixingProton = utils.ndarray2roo(np.array(df_mixingProton[df_column]), x, 'data_mixingProton')

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
    nParams_signal = 1
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
    nParams_signal = 1

  ############### Define uncorrelated bkg function ###############
  if uncorr_bkgPdf == 'pol0':
    uncorr_bkg = ROOT.RooPolynomial("uncorr_bkg", "uncorr_bkg", x, ROOT.RooArgList())
    nParams_uncorr_bkg = 0
  elif uncorr_bkgPdf == 'pol1':
    c1 = ROOT.RooRealVar('c1', 'c1', 3, 0, 1. / (3.04 - lowMassLim) )
    uncorr_bkg = ROOT.RooPolynomial("uncorr_bkg", "uncorr_bkg", xShifted, ROOT.RooArgList(c1))
    nParams_uncorr_bkg = 1
  elif uncorr_bkgPdf == 'pol2':
    pol_a = ROOT.RooRealVar('pol_a', 'pol_a', 3, -30, 30) #0-20
    pol_b = ROOT.RooRealVar('pol_b', 'pol_b', 3, -30, 30) #0-20
    uncorr_bkg = ROOT.RooPolynomial("uncorr_bkg", "uncorr_bkg", xShifted, ROOT.RooArgList(pol_a, pol_b))
    nParams_uncorr_bkg = 2
  elif uncorr_bkgPdf == 'expo':
    c1 = ROOT.RooRealVar('c1', 'c1', -35., -100., -10.)
    uncorr_bkg = ROOT.RooExponential('uncorr_bkg', 'uncorr_bkg', x, c1)
    nParams_uncorr_bkg = 1

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
    nParams_corr_bkg = 3
  elif corr_bkgPdf == "dscb":
    bkg_mu = ROOT.RooRealVar("bkg_mu", "bkg_mu", 2.994, 2.986, 3.01)
    if corr_bkg_peak is not None:
      bkg_mu.setVal(corr_bkg_peak.getVal())
      if fix_bkg_peak:
        bkg_mu.setConstant(ROOT.kTRUE)
      else:
        bkg_mu.setRange(corr_bkg_peak.getVal() - 3*corr_bkg_peak.getError(), corr_bkg_peak.getVal() + 3*corr_bkg_peak.getError())
    bkg_sigma = ROOT.RooRealVar("bkg_sigma", "bkg_sigma", 0.0005, 0.01)
    bkg_a1 = ROOT.RooRealVar("bkg_a_{1}", "bkg_a1", 0.1, 2)
    bkg_n1 = ROOT.RooRealVar("bkg_n_{1}", "bkg_n1", 10, 0.1, 20)
    bkg_a2 = ROOT.RooRealVar("bkg_a_{2}", "bkg_a2", 0.1, 2)
    bkg_n2 = ROOT.RooRealVar("bkg_n_{2}", "bkg_n2", 0.1, 10)
    corr_bkg = ROOT.RooCrystalBall("corr_bkg", "corr_bkg", x, bkg_mu, bkg_sigma, bkg_a1, bkg_n1, bkg_a2, bkg_n2)
    nParams_corr_bkg = 6
  elif corr_bkgPdf == "atan":
    atan_a = ROOT.RooRealVar("atan_a", "atan_a", 0.3, 0.00001, 1000)
    atan_b = ROOT.RooRealVar("atan_b", "atan_b", -1, -1000, 0)
    corr_bkg = ROOT.RooATan("corr_bkg", "corr_bkg", x, atan_a, atan_b)
    nParams_corr_bkg = 2
  elif corr_bkgPdf == "landau":
    landau_m0 = ROOT.RooRealVar("landau_m0", "landau_m0", 2.996, 2.986, 3.01)
    landau_sigma = ROOT.RooRealVar("landau_sigma", "landau_sigma", 0.0005, 0.1)
    corr_bkg = ROOT.RooLandau("corr_bkg", "corr_bkg", x, landau_m0, landau_sigma)
    nParams_corr_bkg = 2
  elif corr_bkgPdf == "pol5":
    bkg_a1 = ROOT.RooRealVar("bkg_a1", "bkg_a1", -1., 1.)
    bkg_a2 = ROOT.RooRealVar("bkg_a2", "bkg_a2", -1., 1.)
    bkg_a3 = ROOT.RooRealVar("bkg_a3", "bkg_a3", -1., 1.)
    bkg_a4 = ROOT.RooRealVar("bkg_a4", "bkg_a4", -1., 1.)
    bkg_a5 = ROOT.RooRealVar("bkg_a5", "bkg_a5", -1., 1.)
    corr_bkg = ROOT.RooChebychev("corr_bkg", "corr_bkg", x, ROOT.RooArgList(bkg_a1, bkg_a2, bkg_a3, bkg_a4, bkg_a5))
    nParams_corr_bkg = 5
  elif corr_bkgPdf == "pol4":
    bkg_a1 = ROOT.RooRealVar("bkg_a1", "bkg_a1", 0.9, -1., 1.)
    bkg_a2 = ROOT.RooRealVar("bkg_a2", "bkg_a2", -0.3,-1., 1.)
    bkg_a3 = ROOT.RooRealVar("bkg_a3", "bkg_a3", -0.3,-1., 1.)
    bkg_a4 = ROOT.RooRealVar("bkg_a4", "bkg_a4", 0.1, -1., 1.)
    corr_bkg = ROOT.RooChebychev("corr_bkg", "corr_bkg", x, ROOT.RooArgList(bkg_a1, bkg_a2, bkg_a3, bkg_a4))
    nParams_corr_bkg = 4
  elif corr_bkgPdf == "pol3":
    bkg_a1 = ROOT.RooRealVar("bkg_a1", "bkg_a1", 0.9, -1., 1.)
    bkg_a2 = ROOT.RooRealVar("bkg_a2", "bkg_a2", -0.3, -1., 1.)
    bkg_a3 = ROOT.RooRealVar("bkg_a3", "bkg_a3", -0.4, -1., 1.)
    corr_bkg = ROOT.RooChebychev("corr_bkg", "corr_bkg", x, ROOT.RooArgList(bkg_a1, bkg_a2, bkg_a3))
    nParams_corr_bkg = 3
  elif corr_bkgPdf == "pol2":
    bkg_a1 = ROOT.RooRealVar("bkg_a1", "bkg_a1", 0.9, -1., 1.)
    bkg_a2 = ROOT.RooRealVar("bkg_a2", "bkg_a2", -0.3, -1., 1.)
    corr_bkg = ROOT.RooChebychev("corr_bkg", "corr_bkg", x, ROOT.RooArgList(bkg_a1, bkg_a2))
    nParams_corr_bkg = 2

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
  fitResult = simPdf.fitTo(combData)
  
  if cfg.isKFAnalysis:
    print('')
    print('')
    print('======================================================')
    print('============= {0} < pT < {1} GeV/c ============='.format(ptlims[0], ptlims[1]))
    print('======================================================')
    print('')

    print('============= Fractions =============')
    if simBkgFit:
      print('frac_corr_mixedDeuteron: ', frac_corr_mixedDeuteron.getVal())
    print('frac_corr: ', frac_corr.getVal())
    print('frac_sig: ', frac_sig.getVal())
    print('')

    print('============= Yield =============')
    print('Raw signal yield: {0} +- {1} '.format(frac_sig.getVal() * data_se.numEntries(), frac_sig.getError() * data_se.numEntries()))
    print('')

    print('============= Signal parameters =============')
    print('mu = {0} + {1}'.format(mu.getVal(), mu.getError()))
    print('sigma = {0} + {1}'.format(sigma.getVal(), sigma.getError()))
    print('')

  ############### Commom script start here ###############

  ############################################
  ############### Draw results ###############
  ############################################

  ############### Draw background spectra ###############
  if cfg.isKFAnalysis:
    canvas_bkg = ROOT.TCanvas("canvas_bkg", "Background", 800, 1500)
  else:
    canvas_bkg = ROOT.TCanvas("canvas_bkg" + title, "Background", 800, 1500)
  pad_uncorr_bkg = ROOT.TPad("pad_uncorr_bkg", "Uncorrelated bkg", 0, 0.5, 1, 1)
  pad_corr_bkg = ROOT.TPad("pad_corr_bkg", "Correlated bkg", 0, 0, 1, 0.5)
  pad_uncorr_bkg.SetBottomMargin(0.02)
  pad_corr_bkg.SetTopMargin(0)
  pad_corr_bkg.SetBottomMargin(0.13)
  pad_uncorr_bkg.Draw()
  pad_corr_bkg.Draw()

  # Uncorrelated background pad_uncorr_bkg
  pad_uncorr_bkg.cd()
  frame_bkg_uncorr = x.frame(RooFit.Bins(nBins))  # Apply binning
  data_mixingProton.plotOn(frame_bkg_uncorr, RooFit.Binning(nBins))
  uncorr_bkg.plotOn(frame_bkg_uncorr, RooFit.LineStyle(9), RooFit.LineColor(utils.kGreenC))
  chi2Val = frame_bkg_uncorr.chiSquare()
  frame_bkg_uncorr.SetTitle('')
  frame_bkg_uncorr.GetXaxis().SetTitle('')
  frame_bkg_uncorr.GetXaxis().SetLabelSize(0)
  frame_bkg_uncorr.GetYaxis().SetTitle(f'Counts / ({bin_width}' + ' GeV/#it{c}^{2})')
  if cfg.isKFAnalysis:
    frame_bkg_uncorr.GetYaxis().SetRangeUser(0, 1.2 * frame_bkg_uncorr.GetMaximum())
  else:
    frame_bkg_uncorr.GetYaxis().SetRangeUser(0, 1.75 * frame_bkg_uncorr.GetMaximum())

  if cfg.isPerformancePlotting:
    if cfg.isKFAnalysis:
      paveText_uncorr = ROOT.TPaveText(0.15, 0.55, 0.56, 0.85, "NDC")
    else:
      paveText_uncorr = ROOT.TPaveText(0.15, 0.5, 0.56, 0.85, "NDC")
    paveText_uncorr.SetName("paveText_uncorr")
    paveText_uncorr.SetBorderSize(0)
    paveText_uncorr.SetFillStyle(0)
    paveText_uncorr.SetTextFont(42)
    paveText_uncorr.SetTextAlign(11)
    paveText_uncorr.SetTextSize(0.045)
    paveText_uncorr.AddText('ALICE Performance')
    paveText_uncorr.AddText('pp #sqrt{#it{s}} = 13.6 TeV')
    paveText_uncorr.AddText('{}_{#Lambda}^{3}H#rightarrow p+#pi+d + cc.')
    paveText_uncorr.AddText('{:.1f}'.format(ptlims[0]) + ' < #it{p}_{T} < ' + '{:.1f}'.format(ptlims[1]) + ' GeV/#it{c}')
    paveText_uncorr.AddText('')
    paveText_uncorr.AddText('Mixed proton/pion background')
  else:
    paveText_uncorr = ROOT.TPaveText(0.15, 0.75, 0.25, 0.85, "NDC")
    paveText_uncorr.AddText(f"#chi^{{2}} = {chi2Val:.2f}")
    paveText_uncorr.SetFillColor(0)
    paveText_uncorr.SetBorderSize(0)
  frame_bkg_uncorr.addObject(paveText_uncorr)
  frame_bkg_uncorr.Draw()

  # Correlated background pad_corr_bkg
  pad_corr_bkg.cd()
  frame_bkg_corr = x.frame(RooFit.Bins(nBins))  # Apply binning
  data_mixingDeuteron.plotOn(frame_bkg_corr, RooFit.Binning(nBins))
  total_bkg_pdf.plotOn(frame_bkg_corr, RooFit.LineStyle(7), RooFit.LineColor(utils.kOrangeC))
  chi2Val = frame_bkg_corr.chiSquare()
  if simBkgFit == True:
    total_bkg_pdf.plotOn(frame_bkg_corr, RooFit.Components("uncorr_bkg"), RooFit.LineStyle(9), RooFit.LineColor(utils.kGreenC))
  frame_bkg_corr.SetTitle('')
  frame_bkg_corr.GetXaxis().SetTitle('#it{M}_{p+#pi+d} (GeV/#it{c}^{2})')
  frame_bkg_corr.GetYaxis().SetTitle(f'Counts / ({bin_width}' + ' GeV/#it{c}^{2})')
  frame_bkg_corr.GetYaxis().SetRangeUser(0, 1.2 * frame_bkg_corr.GetMaximum())

  if cfg.isPerformancePlotting:
    paveText_corr = ROOT.TPaveText(0.15, 0.85, 0.45, 0.98, "NDC")
    # paveText_corr = ROOT.TPaveText(0.15, 0.8, 0.45, 0.98, "NDC")
    paveText_corr.SetName("paveText_corr")
    paveText_corr.SetBorderSize(0)
    paveText_corr.SetFillStyle(0)
    paveText_corr.SetTextFont(42)
    paveText_corr.SetTextAlign(11)
    paveText_corr.SetTextSize(0.045)
    paveText_corr.AddText('Mixed deuteron background')
  else:
    total_bkg_pdf.paramOn(frame_bkg_corr, Layout = [0.9, 0.6, 0.9])
    paveText_corr = ROOT.TPaveText(0.15, 0.25, 0.25, 0.35, "NDC")
    paveText_corr.AddText(f"#chi^{{2}} = {chi2Val:.2f}")
    paveText_corr.SetFillColor(0)
    paveText_corr.SetBorderSize(0)
  frame_bkg_corr.addObject(paveText_corr)
  frame_bkg_corr.Draw()

  ############### Draw total fit ###############
  if cfg.isKFAnalysis:
    canvas_signal = ROOT.TCanvas("canvas_signal", "Fit Results", 800, 1000)
  else:
    canvas_signal = ROOT.TCanvas("canvas_signal" + title, "SE Fit Results", 800, 1000)
  pad1 = ROOT.TPad("pad1", "Fit", 0, 0.3, 1, 1)
  pad2 = ROOT.TPad("pad2", "Residual", 0, 0, 1, 0.3)
  pad1.SetBottomMargin(0.02)
  pad2.SetTopMargin(0)
  pad2.SetBottomMargin(0.3)
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
  total_pdf.plotOn(frame1, RooFit.Components("uncorr_bkg"), RooFit.LineStyle(9), RooFit.LineColor(utils.kGreenC))
  total_pdf.plotOn(frame1, RooFit.Components("total_sig_bkg_pdf"), RooFit.LineStyle(7), RooFit.LineColor(utils.kOrangeC), RooFit.Name("background"))
  total_pdf.plotOn(frame1, RooFit.LineColor(utils.kBlueC))
  if cfg.isKFAnalysis:
    frame1.SetTitle('')
  else:
    frame1.SetTitle(canvas_signal.GetTitle() + " " + title)  # Set title for upper plot
  frame1.GetXaxis().SetTitle("")  # Remove x-axis title in upper plot
  frame1.GetXaxis().SetLabelOffset(999)
  frame1.GetXaxis().SetLabelSize(0)
  frame1.GetYaxis().SetTitle(f'Counts / ({bin_width}' + ' GeV/#it{c}^{2})')
  frame1.GetYaxis().SetTitleSize(0.05)
  frame1.GetYaxis().SetLabelSize(0.04)
  frame1.GetYaxis().SetTitleOffset(1.1)
  frame1.GetYaxis().SetRangeUser(0, frame1.GetMaximum()*1.2)
  frame1.Draw()
  # Info
  if cfg.isPerformancePlotting:
    if cfg.isKFAnalysis:
      paveText = ROOT.TPaveText(0.15, 0.46, 0.52, 0.85, "NDC")
    else:
      paveText = ROOT.TPaveText(0.15, 0.5, 0.52, 0.85, "NDC")
    paveText.SetName("paveText")
    paveText.SetBorderSize(0)
    paveText.SetFillStyle(0)
    paveText.SetTextFont(42)
    paveText.SetTextAlign(11)
    paveText.SetTextSize(0.04)
    paveText.AddText('ALICE Performance')
    paveText.AddText('pp #sqrt{#it{s}} = 13.6 TeV')
    paveText.AddText('#it{L}_{int.} = 21 pb^{-1}')
    paveText.AddText('')
    paveText.AddText('{}_{#Lambda}^{3}H#rightarrow p+#pi+d + cc.')
    paveText.AddText('{:.1f}'.format(ptlims[0]) + ' < #it{p}_{T} < ' + '{:.1f}'.format(ptlims[1]) + ' GeV/#it{c}')
  else:
    paveText = ROOT.TPaveText(0.14, 0.3, 0.5, 0.86, "NDC")
    paveText.AddText('')
    paveText.AddText('#mu = ' + str(round(mu.getValV(), 5)) + ' #pm ' + str(round(mu.getError(), 5)))
    paveText.AddText('#sigma = ' + str(round(sigma.getValV(), 5)) + ' #pm ' + str(round(sigma.getError(), 5)))
    paveText.AddText('')
    chi2Val = frame1.chiSquare()
    # chi2Val = chi2Val*nBins / (nBins*3 - (nParams_signal + nParams_corr_bkg + nParams_uncorr_bkg))
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
  signal.plotOn(frame_residuals, RooFit.LineColor(utils.kRedC), RooFit.LineStyle(ROOT.kSolid), RooFit.Name("signal"))
  frame_residuals.SetTitle("")
  frame_residuals.GetXaxis().SetTitle('#it{M}_{p+#pi+d} (GeV/#it{c}^{2})')
  frame_residuals.GetXaxis().SetTitleSize(0.1)
  frame_residuals.GetXaxis().SetLabelSize(0.1)
  frame_residuals.GetXaxis().SetTitleOffset(1.0)
  frame_residuals.GetYaxis().SetTitle("Residuals to bkg")
  frame_residuals.GetYaxis().SetTitleSize(0.1)
  frame_residuals.GetYaxis().SetLabelSize(0.08)
  frame_residuals.GetYaxis().SetTitleOffset(0.5)
  frame_residuals.SetMarkerStyle(20)
  frame_residuals.SetMarkerSize(0.8)
  frame_residuals.SetLineColor(ROOT.kBlack)
  # Add dashed gray line at y=0
  line = ROOT.TLine(2.96, 0, 3.02, 0)
  line.SetLineColor(ROOT.kGray+2)
  line.SetLineStyle(2)
  line.SetLineWidth(2)
  frame_residuals.addObject(line)
  frame_residuals.Draw()

  # Update canvas
  canvas_signal.Update()
  canvas_signal.RedrawAxis()

  ############### Commom script end here ###############

  if corr_bkgPdf == "dscb":
    bkg_peak_val = bkg_mu
  else:
    bkg_peak_val = None

  return (hRawYield, canvas_bkg, canvas_signal, nSignal, nSignal_err, expNBkg, bkg_peak_val)