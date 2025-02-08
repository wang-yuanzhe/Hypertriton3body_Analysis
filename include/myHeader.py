import numpy as np
import ROOT
import math
import pickle
from copy import deepcopy

ROOT.Math.IntegratorOneDimOptions.SetDefaultIntegrator("GaussLegendre") # To avoid warning

kOrangeC  = ROOT.TColor.GetColor("#ff7f00")

# ****************************************
def createEmptyList(size):
    # for size  = [m,n,...], create a list with dim=m*n*... and return it
    list = []
    for i in reversed(size):
        list = [deepcopy(list) for j in range(i)]
    return list
    
# ****************************************
def gStyleInit():
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1111) 
    ROOT.gStyle.SetTitleSize(0.05,"X")
    ROOT.gStyle.SetTitleSize(0.05,"Y")
    ROOT.gStyle.SetLabelSize(0.04,"X")
    ROOT.gStyle.SetLabelSize(0.04,"Y")

# ****************************************
def calVirtualLambdaInvM(proton_pt, proton_eta, proton_phi, pion_pt, pion_eta, pion_phi):
    proton_px = proton_pt * np.cos(proton_phi)
    proton_py = proton_pt * np.sin(proton_phi)
    proton_pz = proton_pt * np.sinh(proton_eta)

    pion_px = pion_pt * np.cos(pion_phi)
    pion_py = pion_pt * np.sin(pion_phi)
    pion_pz = pion_pt * np.sinh(pion_eta)

    proton_mass = 0.9382721
    pion_mass = 0.1395704

    proton_e = np.sqrt(proton_px**2 + proton_py**2 + proton_pz**2 + proton_mass**2)
    pion_e = np.sqrt(pion_px**2 + pion_py**2 + pion_pz**2 + pion_mass**2)

    lambda_e = proton_e + pion_e
    lambda_px = proton_px + pion_px
    lambda_py = proton_py + pion_py
    lambda_pz = proton_pz + pion_pz

    lambda_m = np.sqrt(lambda_e**2 - lambda_px**2 - lambda_py**2 - lambda_pz**2)

    return lambda_m
# ****************************************
def calMdp(proton_pt, proton_eta, proton_phi, deuteron_pt, deuteron_eta, deuteron_phi):
    proton_px = proton_pt * np.cos(proton_phi)
    proton_py = proton_pt * np.sin(proton_phi)
    proton_pz = proton_pt * np.sinh(proton_eta)

    deuteron_px = deuteron_pt * np.cos(deuteron_phi)
    deuteron_py = deuteron_pt * np.sin(deuteron_phi)
    deuteron_pz = deuteron_pt * np.sinh(deuteron_eta)

    proton_mass = 0.9382721
    deuteron_mass = 1.87561294257

    proton_e = np.sqrt(proton_px**2 + proton_py**2 + proton_pz**2 + proton_mass**2)
    deuteron_e = np.sqrt(deuteron_px**2 + deuteron_py**2 + deuteron_pz**2 + deuteron_mass**2)

    total_e = proton_e + deuteron_e
    total_px = proton_px + deuteron_px
    total_py = proton_py + deuteron_py
    total_pz = proton_pz + deuteron_pz

    invariant_mass = np.sqrt(total_e**2 - total_px**2 - total_py**2 - total_pz**2)
    return invariant_mass

# ****************************************
def calNewElements(df):
    df['fDiffRDaughter'] = df['fRadiusBachelor'] - df[['fRadiusProton', 'fRadiusPion']].min(axis=1)
    df['fDiffRProton'] = df['fRadiusProton'] - df['fVtxRadius']
    df['fDiffRPion'] = df['fRadiusPion'] - df['fVtxRadius']
    df['fDiffRBachelor'] = df['fRadiusBachelor'] - df['fVtxRadius']
    df['fMVirtualLambda'] = calVirtualLambdaInvM(df['fPtProton'], df['fEtaProton'], df['fPhiProton'], df['fPtPion'], df['fEtaPion'], df['fPhiPion'])
    df['fMdp'] = calMdp(df['fPtProton'], df['fEtaProton'], df['fPhiProton'], df['fPtBachelor'], df['fEtaBachelor'], df['fPhiBachelor'])

# ****************************************
def fitInvMass(df, title, massbin = [2.96, 3.04], nbins = 40, sigpdf = "Gauss", bkgpdf = "pol1", para = None, isMC = False, ifDrawStats = True, ifDebug = False, Matter = 0):
         
    if not(sigpdf in ("Gauss", "DSCB", "KDE")):
        raise Exception("Undefined sigpdf!")
    if not(bkgpdf in ("none", "pol1", "pol2", "exp", "Argus", "doubleBkg")):
        raise Exception("Undefined bkgpdf!")

    if not para:
        x = ROOT.RooRealVar("x", "x", massbin[0], massbin[-1])
    else:
        x = para[-1]
    
    #nbins = 0 -> unbinned fit
    if len(df)==0:
        n_signal = ROOT.RooRealVar('N_{signal}', 'Nsignal', 0, 0)
        n_bkg = ROOT.RooRealVar('N_{bkg}', 'Nbackground', 0, 0)
        para = False
        bkgcount = 0
        xframe = x.frame(ROOT.RooFit.Title(title), ROOT.RooFit.Name("dataframe"))
        return (xframe, n_signal, bkgcount, para)
    
    if nbins == 0:
        data = ndarray2roo(np.array(df['fM']),x)
    else:
        h1 = ROOT.TH1D("h1", "h1", nbins, massbin[0], massbin[-1])
        h = (massbin[-1] - massbin[0])/nbins
        for i in range(nbins):
            h1.SetBinContent(i+1, sum(np.logical_and(df['fM'] >= massbin[0] + i*h, df['fM'] < massbin[0] + (i+1)*h)) )
        data = ROOT.RooDataHist("data", "dataset with x", x, h1)

    # Set signal fit function
    mu = ROOT.RooRealVar("#mu", "mu", 2.991, 2.986, 2.996)
    sigma = ROOT.RooRealVar("#sigma", "sigma", 0.0005, 0.0030)

    if sigpdf=="DSCB":
        a1 = ROOT.RooRealVar("a_{1}", "a1", 0.1, 2)
        n1 = ROOT.RooRealVar("n_{1}", "n1", 0.1, 10)
        a2 = ROOT.RooRealVar("a_{2}", "a2", 0.1, 2)
        n2 = ROOT.RooRealVar("n_{2}", "n2", 0.1, 10)
        if para:
            sigma.setVal(para[1].getVal())
            a1.setVal(para[2].getVal())
            n1.setVal(para[3].getVal())
            a2.setVal(para[4].getVal())
            n2.setVal(para[5].getVal())
            sigma.setConstant(ROOT.kTRUE)
            a1.setConstant(ROOT.kTRUE)
            n1.setConstant(ROOT.kTRUE)
            a2.setConstant(ROOT.kTRUE)
            n2.setConstant(ROOT.kTRUE)
        fit_signal = ROOT.RooCrystalBall("fit_signal", "fit_signal", x, mu, sigma, a1, n1, a2, n2)
    elif sigpdf=="Gauss":
        if para:
            sigma.setVal(para[1].getVal())
            sigma.setConstant(ROOT.kTRUE)
        fit_signal = ROOT.RooGaussian("fit_signal", "fit_signal", x, mu, sigma)
    elif sigpdf=="KDE":
        if para:
            fit_signal = para[0].Clone("fit_signal")
        else:
            fit_signal = ROOT.RooKeysPdf("fit_signal", "fit_signal", x, data, ROOT.RooKeysPdf.MirrorBoth, 2)
    
    # Set background fit function
    if (isMC == True) or bkgpdf == "none":
        bkg = ROOT.RooPolynomial('bkg','bkg', x, ROOT.RooArgList(ROOT.RooFit.RooConst(0)))
    elif bkgpdf == "exp":
        c1 = ROOT.RooRealVar('c1', 'c1', -20., -0.0001)
        bkg = ROOT.RooExponential('bkg', 'bkg', x, c1)
    elif bkgpdf == "pol1":
        c1 = ROOT.RooRealVar('c1', 'c1', -20., 20.)
        bkg = ROOT.RooPolynomial("bkg", "bkg", x, ROOT.RooArgList(c1))
    elif bkgpdf == "pol2":
        c1 = ROOT.RooRealVar('c1', 'c1', 0, -20., 20.)
        c2 = ROOT.RooRealVar('c2', 'c2', -10, -100, 0)
        bkg = ROOT.RooPolynomial("bkg", "bkg", x, ROOT.RooArgList(c1, c2))
    elif bkgpdf == "Argus":
        xMirrored = ROOT.RooLinearVar("x_mirrored", "x_mirrored", x, ROOT.RooFit.RooConst(-1.0), ROOT.RooFit.RooConst(3.08))
        argus_m0 = ROOT.RooRealVar("argus_m0", "argus_m0", 0.12, 0.10, 0.18)
        argus_c = ROOT.RooRealVar("argus_c", "argus_c", -5, -10, -1)
        argus_p = ROOT.RooRealVar("argus_p", "argus_p", 3, 0.5, 5)
        bkg = ROOT.RooArgusBG("bkg", "bkg", xMirrored, argus_m0, argus_c, argus_p)
    elif bkgpdf == "doubleBkg":
        c1 = ROOT.RooRealVar('c1', 'c1', 0.1, 10)
        c2 = ROOT.RooRealVar('c2', 'c2', 10, -20, 20.)
        xMirrored = ROOT.RooLinearVar("x_mirrored", "x_mirrored", x, ROOT.RooFit.RooConst(-1.0), ROOT.RooFit.RooConst(3.08))
        argus_m0 = ROOT.RooRealVar("argus_m0", "argus_m0", 0.143, 0.10, 0.18)
        argus_c = ROOT.RooRealVar("argus_c", "argus_c", -5, -10, -1)
        argus_p = ROOT.RooRealVar("argus_p", "argus_p", 3, 0.5, 5)
        bkg1 = ROOT.RooArgusBG("bkg1", "bkg1", xMirrored, argus_m0, argus_c, argus_p) 
        # bkg2 = ROOT.RooPolynomial("bkg2", "bkg2", x, ROOT.RooArgList(c1))
        bkg2 = ROOT.RooPolynomial("bkg2", "bkg2", x)
        # s1 = ROOT.RooRealVar('s1', 's1',  0.0001, 10)
        # bkg2 = ROOT.RooExponential("bkg2", "bkg2", xMirrored, s1)


    # Plot the data
    xframe = x.frame(ROOT.RooFit.Title(title), ROOT.RooFit.Name("dataframe"))
    data.plotOn(xframe, ROOT.RooFit.Binning(32))
    
    # Fit the data and extract the yield
    n_signal = ROOT.RooRealVar('N_{signal}', 'Nsignal', 0, len(df))
    if bkgpdf == "doubleBkg":
        n_bkg1 = ROOT.RooRealVar('N_{bkg1}', 'Nbackground1', 0, len(df))
        n_bkg2 = ROOT.RooRealVar('N_{bkg2}', 'Nbackground2', 0, len(df))
        model = ROOT.RooAddPdf("model","model", ROOT.RooArgList(fit_signal, bkg1, bkg2), ROOT.RooArgList(n_signal, n_bkg1, n_bkg2))
    else:
        n_bkg = ROOT.RooRealVar('N_{bkg}', 'Nbackground', 0, len(df))
        model = ROOT.RooAddPdf("model","model", ROOT.RooArgList(fit_signal, bkg), ROOT.RooArgList(n_signal, n_bkg))
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ROOT.RooMsgService.instance().setSilentMode(ROOT.kTRUE)

    #To avoid unexpected pdf, see the tutorial rf612_recoverFromInvalidParameters.py
    r = model.fitTo(data, ROOT.RooFit.Save(), ROOT.RooFit.Extended(ROOT.kTRUE), Save=True,
    RecoverFromUndefinedRegions=1.0,  # The magnitude of the recovery information can be chosen here. Higher values mean more aggressive recovery.
    PrintEvalErrors=-1,
    PrintLevel=-1,
    )

    if (isMC==False):
        if bkgpdf == "doubleBkg":
            model.plotOn(xframe, ROOT.RooFit.Components("bkg1,bkg2"),
                            ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineColor(kOrangeC))
        else:
            model.plotOn(xframe, ROOT.RooFit.Components('bkg'),ROOT.RooFit.Name('bkg'),
                         ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineColor(kOrangeC))
        #model.plotOn(xframe, ROOT.RooFit.Components('fit_signal'), ROOT.RooFit.Name('fit_signal'), ROOT.RooFit.LineColor(ROOT.kRed))
    cmodel = model.plotOn(xframe, ROOT.RooFit.Name('model'), ROOT.RooFit.LineColor(ROOT.kAzure + 2)).getCurve()
    xframe.addObject(cmodel)
    if (Matter == -1):
        xframe.GetXaxis().SetTitle( 'm(#bar{p}+#pi^{-}+#bar{d}) (GeV/c^{2})' )
    elif (Matter == 1): 
        xframe.GetXaxis().SetTitle( 'm(p+#pi^{+}+d) (GeV/c^{2})' )
    else:
        xframe.GetXaxis().SetTitle( 'm(p+#pi+d) (GeV/c^{2})' )
    xframe.GetYaxis().SetTitle(xframe.GetYaxis().GetTitle()[:-1] + 'GeV/c^{2} )')
    xframe.Draw()

    if (isMC==True) or (ifDebug == True):
        model.paramOn(xframe, Layout = [0.2, 0.4, 0.9])

    bkgcount = 0
    x.setRange("signal", mu.getVal() - 3 * sigma.getVal(), mu.getVal() + 3 * sigma.getVal())
    if (isMC == False) :
        if bkgpdf == "doubleBkg":
            bkg1Normcount = model.pdfList().at(1).createIntegral(ROOT.RooArgSet(x),ROOT.RooFit.NormSet(x),ROOT.RooFit.Range("signal"))
            bkg2Normcount = model.pdfList().at(2).createIntegral(ROOT.RooArgSet(x),ROOT.RooFit.NormSet(x),ROOT.RooFit.Range("signal"))
            bkgcount = n_bkg1.getVal() * bkg1Normcount.getVal() + n_bkg2.getVal() * bkg2Normcount.getVal()
            unc_bkg = n_bkg1.getError() * bkg1Normcount.getVal() + n_bkg2.getError() * bkg2Normcount.getVal()
        else:
            bkgNormcount = model.pdfList().at(1).createIntegral(ROOT.RooArgSet(x),ROOT.RooFit.NormSet(x),ROOT.RooFit.Range("signal"))
            bkgcount = bkgNormcount.getVal() * n_bkg.getVal()
            unc_bkg = n_bkg.getError()*bkgNormcount.getVal()

    sigNormcount = model.pdfList().at(0).createIntegral(ROOT.RooArgSet(x),ROOT.RooFit.NormSet(x),ROOT.RooFit.Range("signal"))
    sigcount = sigNormcount.getVal() * n_signal.getVal()
    unc_sig = n_signal.getError()*sigNormcount.getVal()
    
    if ifDrawStats:   
        paveText = ROOT.TPaveText(0.55, 0.6, 0.9, 0.9, "NDC")
        paveText.SetName("paveText")
        paveText.SetBorderSize(0)
        paveText.SetFillStyle(0)
        paveText.SetTextFont(42)
        paveText.SetTextAlign(11)
        # paveText.SetTextSize(28)
        paveText.AddText('S = ' + str(round(n_signal.getVal())) + ' #pm ' + str(round(n_signal.getError())))
        paveText.AddText('B(3#sigma) = ' + str(round(bkgcount)) + ' #pm ' + str(round(unc_bkg)))
        if (isMC == False):
            if (bkgcount > 0):
                unc_significance = math.sqrt( math.pow(unc_sig / bkgcount, 2) + math.pow( unc_bkg * sigcount / (bkgcount * bkgcount), 2) )
                paveText.AddText('S/B(3#sigma) = ' + str( round(sigcount / bkgcount, 2) ) + ' #pm ' + str(round(unc_significance, 2)))
            else:
                paveText.AddText('S/B(3#sigma) = NaN') 
            if (n_signal.getVal() + bkgcount > 0):
                unc_significance = math.sqrt( math.pow( unc_sig * (sigcount + 2*bkgcount) / (2 * (sigcount + bkgcount) * math.sqrt(sigcount + bkgcount)), 2) + math.pow( unc_bkg * sigcount / (2 * math.pow(sigcount + bkgcount, 1.5)), 2) )
                paveText.AddText('Significance(3#sigma) = ' + str( round(sigcount/(math.sqrt(sigcount + bkgcount)) ,1) ) + ' #pm ' + str(round(unc_significance, 1)) )
            else:
                paveText.AddText('Significance(3#sigma) = NaN')
            # paveText.AddText('#mu = ' + str(round(mu.getValV(), 5)) + ' #pm ' + str(round(mu.getError(), 5)))
            # paveText.AddText('#sigma = ' + str(round(sigma.getValV(), 5)) + ' #pm ' + str(round(sigma.getError(), 5)))
        xframe.addObject(paveText)
        paveText.Draw()
    
    para = []
    para.extend([mu,sigma])
    if sigpdf == "DSCB":
        para.extend([a1,n1,a2,n2])
    if sigpdf == "KDE":
        para = [fit_signal]
    para.append(x)
    return (xframe, n_signal, bkgcount, para)

# ****************************************
def ndarray2roo(ndarray, var):
    if isinstance(ndarray, ROOT.RooDataSet):
        print('Already a RooDataSet')
        return ndarray

    assert isinstance(ndarray, np.ndarray), 'Did not receive NumPy array'
    assert len(ndarray.shape) == 1, 'Can only handle 1d array'

    name = var.GetName()
    x = np.zeros(1, dtype=np.float64)

    tree = ROOT.TTree('tree', 'tree')
    tree.Branch(f'{name}', x, f'{name}/D')

    for i in ndarray:
        x[0] = i
        tree.Fill()

    array_roo = ROOT.RooDataSet(
        'data', 'dataset from tree', tree, ROOT.RooArgSet(var))
    return array_roo

# ****************************************
def convert_sel_to_string(selection):
    sel_string = ''
    conj = ' and '
    for sel in selection:
        for _, val in sel.items():
            sel_string = sel_string + val + conj
    return sel_string[:-len(conj)]

# ****************************************
def apply_pt_rejection(df, pt_shapeList, cent_bin_list, pt_bin_list, option="Default", path=""):
    #Reweight hypertriton spectrum for MC dataset
    if option.upper() == "READ":
        rej_flag = pickle.load(open(path, "rb"))
    else:
        rej_flag = np.ones(len(df), dtype=bool)
        random_arr = np.random.rand(len(df))
        
        for ind, (centrality, ptMC, rand) in enumerate(zip(df['fCentrality'], df['fGenPt'], random_arr)):
            for centbin, ptbins, pt_shape in zip(cent_bin_list, pt_bin_list, pt_shapeList):
                if centrality >= centbin[0] and centrality < centbin[1]:
                    for ptbin in ptbins:
                        if ptMC >= ptbin[0] and ptMC < ptbin[1]:
                            frac = pt_shape.Eval(ptMC)/pt_shape.GetMaximum(ptbin[0], ptbin[1])
                            if rand < frac:
                                rej_flag[ind] = False
                            continue
    if option.upper() =="SAVE" and (path != ""):
        pickle.dump(rej_flag, open(path, "wb"))
    df["rej"]=rej_flag

# ****************************************
def SetTCanvas(c):
    c.SetBorderMode(0)
    c.SetFillColor(10)
    c.SetFrameFillColor(0)
    c.SetFrameBorderMode(0)
    c.SetLeftMargin(0.15)
    c.SetTopMargin(0.08)
    c.SetBottomMargin(0.15)
    c.SetRightMargin(0.08)

# ****************************************
def SetTLegend(legend):
    legend.SetName("leg")
    legend.SetBorderSize(0)
    legend.SetTextSize(0.06)
    legend.SetLineColor(0)
    legend.SetLineStyle(0)
    legend.SetLineWidth(0)
    legend.SetFillColor(ROOT.kBlack)
    legend.SetFillStyle(3)

# ****************************************
def TCanvas(*args, **kwargs):
    C = ROOT.TCanvas(*args, **kwargs)
    SetTCanvas(C)
    return C