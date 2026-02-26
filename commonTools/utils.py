import numpy as np
import ROOT
import math
import pickle
import uproot
import pandas as pd
import fnmatch
import re
from copy import deepcopy

ROOT.Math.IntegratorOneDimOptions.SetDefaultIntegrator("GaussLegendre") # To avoid warning

kBlueC = ROOT.kAzure-8
kOrangeC = ROOT.kOrange-3
kGreenC = ROOT.kSpring+3
kRedC = ROOT.kRed+1

# ****************************************
def fitInvMass(df, title, massbin = [2.96, 3.04], nbins = 40, sigpdf = "Gauss", bkgpdf = "pol1", para = None, isMC = False, ifDrawStats = True, ifDebug = False, Matter = 0, col = "fM", mu_range = [2.991, 2.986, 2.996], sigma_range = [0.00175, 0.0005, 0.003], xtitle = None):
         
    if not(sigpdf.lower() in ("gauss", "dscb", "kde", "landau")):
        raise Exception("Undefined sigpdf!")
    if not(bkgpdf.lower() in ("none", "pol1", "pol2", "exp", "argus", "doublebskg")):
        raise Exception("Undefined bkgpdf!")

    # if para == None:
    #     x = ROOT.RooRealVar("x", "x", massbin[0], massbin[-1])
    # else:
    #     x = para[-1]

    x = ROOT.RooRealVar("x", "x", massbin[0], massbin[-1]) # check: what will happen?
    fitdf = df.query(f"{col} >= {massbin[0]} and {col} < {massbin[-1]}")
    
    #nbins = 0 -> unbinned fit
    if len(fitdf)==0:
        n_signal = ROOT.RooRealVar('N_{signal}', 'Nsignal', 0, 0)
        n_bkg = ROOT.RooRealVar('N_{bkg}', 'Nbackground', 0, 0)
        para = None
        bkgcount = 0
        xframe = x.frame(ROOT.RooFit.Title(title), ROOT.RooFit.Name("dataframe"))
        return (xframe, n_signal, bkgcount, para)
    
    if nbins == 0:
        data = ndarray2roo(np.array(fitdf[col]), x)
    else:
        h1 = ROOT.TH1D("h1", "h1", nbins, massbin[0], massbin[-1])
        h = (massbin[-1] - massbin[0])/nbins
        for i in range(nbins):
            h1.SetBinContent(i+1, sum(np.logical_and(fitdf[col] >= massbin[0] + i*h, fitdf[col] < massbin[0] + (i+1)*h)) )
        data = ROOT.RooDataHist("data", "dataset with x", x, h1)

    # Set signal fit function
    mu = ROOT.RooRealVar("#mu", "mu", mu_range[0], mu_range[1], mu_range[2])
    sigma = ROOT.RooRealVar("#sigma", "sigma", sigma_range[0], sigma_range[1], sigma_range[2])

    if sigpdf=="DSCB":
        a1 = ROOT.RooRealVar("a_{1}", "a1", 0.1, 2)
        n1 = ROOT.RooRealVar("n_{1}", "n1", 0.1, 10)
        a2 = ROOT.RooRealVar("a_{2}", "a2", 0.1, 2)
        n2 = ROOT.RooRealVar("n_{2}", "n2", 0.1, 10)
        if para != None:
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
        if para != None:
            sigma.setVal(para[1].getVal())
            sigma.setConstant(ROOT.kTRUE)
        fit_signal = ROOT.RooGaussian("fit_signal", "fit_signal", x, mu, sigma)
    elif sigpdf=="KDE":
        if para != None:
            fit_signal = para[0].Clone("fit_signal")
        else:
            fit_signal = ROOT.RooKeysPdf("fit_signal", "fit_signal", x, data, ROOT.RooKeysPdf.MirrorBoth, 2)
    elif sigpdf == "Landau":
        if para != None:
            sigma.setVal(para[1].getVal())
            sigma.setConstant(ROOT.kTRUE)
        fit_signal = ROOT.RooLandau("fit_signal", "fit_signal", x, mu, sigma)
    
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
    if nbins == 0:
        data.plotOn(xframe, ROOT.RooFit.Binning(32))
    else:
        data.plotOn(xframe, ROOT.RooFit.Binning(nbins))
    
    # Fit the data and extract the yield
    n_signal = ROOT.RooRealVar('N_{signal}', 'Nsignal', 0, len(fitdf))
    if bkgpdf == "doubleBkg":
        n_bkg1 = ROOT.RooRealVar('N_{bkg1}', 'Nbackground1', 0, len(fitdf))
        n_bkg2 = ROOT.RooRealVar('N_{bkg2}', 'Nbackground2', 0, len(fitdf))
        model = ROOT.RooAddPdf("model","model", ROOT.RooArgList(fit_signal, bkg1, bkg2), ROOT.RooArgList(n_signal, n_bkg1, n_bkg2))
    else:
        n_bkg = ROOT.RooRealVar('N_{bkg}', 'Nbackground', 0, len(fitdf))
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
    model.plotOn(xframe, ROOT.RooFit.Name('model'), ROOT.RooFit.LineColor(ROOT.kAzure + 2))
    if (Matter == -1):
        xframe.GetXaxis().SetTitle( 'm(#bar{p}+#pi^{-}+#bar{d}) (GeV/c^{2})' )
    elif (Matter == 1): 
        xframe.GetXaxis().SetTitle( 'm(p+#pi^{+}+d) (GeV/c^{2})' )
    else:
        xframe.GetXaxis().SetTitle( 'm(p+#pi+d) (GeV/c^{2})' )
    if xtitle is not None:
        xframe.GetXaxis().SetTitle(xtitle)
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
            unc_bkg = n_bkg.getError() * bkgNormcount.getVal()

    sigNormcount = model.pdfList().at(0).createIntegral(ROOT.RooArgSet(x),ROOT.RooFit.NormSet(x),ROOT.RooFit.Range("signal"))
    sigcount = sigNormcount.getVal() * n_signal.getVal()
    unc_sig = n_signal.getError() * sigNormcount.getVal()
    
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
    
    paras = []
    paras.extend([mu,sigma])
    if sigpdf == "DSCB":
        paras.extend([a1,n1,a2,n2])
    if sigpdf == "KDE":
        paras = [fit_signal]
    paras.append(x)
    return (xframe, n_signal, bkgcount, paras)

# ****************************************
def fitH3LKinkInvMass(df, title, massbin = [2.94, 3.2], nbins = 0, sigpdf = "Landau", bkgpdf = "none", para = None, isMC = False, ifDrawStats = True, ifDebug = False, Matter = 0, col = "fM", mu_range = [2.991, 2.98, 3], sigma_range = [0.01, 0.001, 0.1], **kwargs):
    return fitInvMass(df, title, massbin, nbins, sigpdf, bkgpdf, para, isMC, ifDrawStats, ifDebug, Matter, col, mu_range, sigma_range, **kwargs)

# ****************************************
def fitHe4SKinkInvMass(df, title, massbin = [3.9, 4.2], nbins = 0, sigpdf = "Landau", bkgpdf = "none", para = None, isMC = False, ifDrawStats = True, ifDebug = False, Matter = 0, col = "fM", mu_range = [3.995, 3.95, 4.05], sigma_range = [0.01, 0.001, 0.1], **kwargs):
    return fitInvMass(df, title, massbin, nbins, sigpdf, bkgpdf, para, isMC, ifDrawStats, ifDebug, Matter, col, mu_range, sigma_range, **kwargs)

# ****************************************
# General utility functions
# ****************************************
def createEmptyList(size):
    # for size  = [m,n,...], create a list with dim=m*n*... and return it
    list = []
    for i in reversed(size):
        list = [deepcopy(list) for j in range(i)]
    return list

# ****************************************
def get_latest_keys(keys):
    key_dict = {}
    pattern = re.compile(r"^(.*);(\d+)$")
    for key in keys:
        match = pattern.match(key)
        if not match:
            continue
        name, cycle = match.group(1), int(match.group(2))
        if name not in key_dict or cycle > key_dict[name][1]:
            key_dict[name] = (key, cycle)
    return [v[0] for v in key_dict.values()]

# ****************************************
def normalize_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi

# ****************************************
def convert_sel_to_string(cfg):
    sel_string = ''
    conj = ' and '
    for sel in cfg:
        for _, val in sel.items():
            sel_string = sel_string + val + conj
    return sel_string[:-len(conj)]

# ****************************************
def load_active_selections(cfg):
    active_selections = []

    for key, val in cfg.items():
        if isinstance(val, dict):
            if val.get("enable", False):
                selection = val.get("value") or val.get("condition")
                if selection:
                    active_selections.append(selection)
        elif isinstance(val, str):
            active_selections.append(val)

    return " and ".join(active_selections)

# ****************************************
class EfficientReplacer:
    def __init__(self, replacement_dict):
        self.pattern = re.compile('|'.join(map(re.escape, replacement_dict.keys())))
        self.replacement_dict = replacement_dict
    
    def replace(self, text):
        return self.pattern.sub(
            lambda match: self.replacement_dict[match.group(0)], 
            text
        )
    
    def __call__(self, text):
        return self.replace(text)
    
# ****************************************
def get_threshold(text, pattern, operator=None):
    if operator is None:
        op_regex = r'[<>]=?|==|!='
    else:
        op_regex = re.escape(operator)
    
    regex = rf'{pattern}\s*({op_regex})\s*([-\d\.]+)'
    match = re.search(regex, text)
    
    if match:
        return float(match.group(2))
    return None

# ****************************************
def update_threshold(text, pattern, new_value, operator=None):
    if operator is None:
        op_regex = r'[<>]=?|==|!='
    else:
        op_regex = re.escape(operator)
    
    pattern = rf'({re.escape(pattern)})\s*({op_regex})\s*([-\d\.]+(?:[eE][-+]?\d+)?)'
    matches = list(re.finditer(pattern, text))
    if len(matches) == 0:
        return text
    elif len(matches) > 1:
        raise ValueError(f"More than one match found for '{pattern}'")

    match = matches[0]
    replacement = f"{match.group(1)} {match.group(2)} {new_value}"
    return text[:match.start()] + replacement + text[match.end():]

# ****************************************
def convert_ptbin_to_dir(PT_BIN_LIST):
    path = "pT"
    for ptbin in PT_BIN_LIST:  # only one centrality bin
        path = f"{path}{ptbin[0]}_"
    path = f"{path}{PT_BIN_LIST[-1][-1]}"
    return path

# ****************************************
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# ****************************************
# Operation on the dataframe
# ****************************************
def getDF_fromFile(file_name, tree_name, folder_name='DF*', **kwds):
    df_list = []
    files = file_name if isinstance(file_name, list) else [file_name]

    for file in files:
        file_obj = uproot.open(file)
        all_keys = get_latest_keys(file_obj.keys())
        tree_paths = [
            key for key in all_keys
            if fnmatch.fnmatch(key, f"{folder_name}/{tree_name}*")
        ]

        for tree_path in tree_paths:
            full_path = f"{file}:{tree_path}"
            tree = uproot.open(full_path)
            df = tree.arrays(filter_name=None, library="pd", **kwds)
            df_list.append(df)

    return pd.concat(df_list, ignore_index=True, copy=False)

# ****************************************
# def ndarray2roo(ndarray, var, data_name='data'):
#     if isinstance(ndarray, ROOT.RooDataSet):
#         print('Already a RooDataSet')
#         return ndarray

#     assert isinstance(ndarray, np.ndarray), 'Did not receive NumPy array'
#     assert len(ndarray.shape) == 1, 'Can only handle 1d array'

#     name = var.GetName()
#     x = np.zeros(1, dtype=np.float64)

#     tree = ROOT.TTree('tree', 'tree')
#     tree.Branch(f'{name}', x, f'{name}/D')

#     for i in ndarray:
#         x[0] = i
#         tree.Fill()

#     array_roo = ROOT.RooDataSet(
#         f'{data_name}', 'dataset from tree', tree=tree, vars=ROOT.RooArgSet(var))
#     return array_roo

def ndarray2roo(ndarray, var, data_name='data'):
    """Convert a 1D numpy array into a RooDataSet using a TTree as intermediate."""

    # Case 1: already a RooDataSet
    if isinstance(ndarray, ROOT.RooDataSet):
        print('Already a RooDataSet')
        return ndarray

    # Type & shape checks
    assert isinstance(ndarray, np.ndarray), 'Did not receive NumPy array'
    assert len(ndarray.shape) == 1, 'Can only handle 1D array'

    # Create TTree
    name = var.GetName()
    x = np.zeros(1, dtype=np.float64)

    tree = ROOT.TTree('tree', 'tree')
    tree.Branch(name, x, f'{name}/D')

    for val in ndarray:
        x[0] = val
        tree.Fill()

    # Correct + future-proof RooDataSet constructor (no keyword args!)
    array_roo = ROOT.RooDataSet(
        data_name,
        "dataset from tree",
        ROOT.RooArgSet(var),
        ROOT.RooFit.Import(tree)
    )

    return array_roo

# ****************************************
def apply_pt_rejection(df, pt_shapeList, cent_bin_list, pt_bin_list, ptcolumn="fGenPt", option="Default", path=""):
    #Reweight hypertriton spectrum for MC dataset
    if option.upper() == "READ":
        rej_flag = pickle.load(open(path, "rb"))
    else:
        rej_flag = np.ones(len(df), dtype=bool)
        random_arr = np.random.rand(len(df))
        
        for ind, (centrality, pt, rand) in enumerate(zip(df['fCentrality'], df[ptcolumn], random_arr)):
            for centbin, ptbins, pt_shape in zip(cent_bin_list, pt_bin_list, pt_shapeList):
                if centrality >= centbin[0] and centrality < centbin[1]:
                    for ptbin in ptbins:
                        if pt >= ptbin[0] and pt < ptbin[1]:
                            frac = pt_shape.Eval(pt)/pt_shape.GetMaximum(ptbin[0], ptbin[1])
                            if rand < frac:
                                rej_flag[ind] = False
                            continue
    if option.upper() =="SAVE" and (path != ""):
        pickle.dump(rej_flag, open(path, "wb"))
    df["rej"]=rej_flag

# ****************************************
def get_peak_value(df, col, bin, output=None):
    hist = ROOT.TH1D("h1", "h1", bin[0], bin[1], bin[2])
    for x in df[col]:
        hist.Fill(x)
    x = ROOT.RooRealVar("x", "Observable", bin[1], bin[2])
    data = ROOT.RooDataHist("data", "dataset from TH1F", ROOT.RooArgList(x), hist)

    mu    = ROOT.RooRealVar("mu",    "mean",  bin[1], bin[2])
    sigma = ROOT.RooRealVar("sigma", "sigma", hist.GetRMS()*0.5, 1e-3, 10*hist.GetRMS())
    a1 = ROOT.RooRealVar("a1", "alpha left",  1.5,  0.01, 10.0)
    n1 = ROOT.RooRealVar("n1", "n left",      2.0,  0.1, 200.0)
    a2 = ROOT.RooRealVar("a2", "alpha right", 1.5,  0.01, 10.0)
    n2 = ROOT.RooRealVar("n2", "n right",     2.0,  0.1, 200.0)

    dscb = ROOT.RooCrystalBall("fit_signal", "fit_signal", x, mu, sigma, a1, n1, a2, n2)
    dscb.fitTo(data, ROOT.RooFit.Save())

    if output is not None:
        c = TCanvas("c", "fit", 800, 600)
        frame = x.frame()
        data.plotOn(frame)
        dscb.plotOn(frame)
        dscb.paramOn(frame, Layout = [0.2, 0.6, 0.5])
        frame.Draw()
        c.SaveAs(output)
    return mu.getVal()

# ****************************************
# Plotting style
# ****************************************
def set_style():
    # ROOT.gStyle.SetOptStat(0)
    # ROOT.gStyle.SetOptFit(1111) 
    ROOT.gStyle.SetPalette(1)
    ROOT.gStyle.SetOptStat(1)
    ROOT.gStyle.SetOptDate(0)
    ROOT.gStyle.SetOptFit(1)
    ROOT.gStyle.SetLabelSize(0.04, 'xyz')
    ROOT.gStyle.SetTitleSize(0.05, 'xyz')
    ROOT.gStyle.SetTitleFont(42, 'xyz')
    ROOT.gStyle.SetLabelFont(42, 'xyz')
    ROOT.gStyle.SetTitleOffset(1.05, 'x')
    ROOT.gStyle.SetTitleOffset(1.1, 'y')
    ROOT.gStyle.SetCanvasDefW(500)
    ROOT.gStyle.SetCanvasDefH(600)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetPadLeftMargin(0.12)
    ROOT.gStyle.SetPadRightMargin(0.10)
    ROOT.gStyle.SetPadGridX(0)
    ROOT.gStyle.SetPadGridY(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetPaperSize(20, 24)
    ROOT.gStyle.SetLegendBorderSize(0)
    ROOT.gStyle.SetLegendFillColor(0)
    ROOT.gStyle.SetEndErrorSize(0.)
    ROOT.gStyle.SetMarkerSize(1)

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

# ****************************************
def plot_hist(df, colx, bins, file, name = None, xtitle = None, ytitle = None):
    file.cd()
    h1 = ROOT.TH1F(f"h_{colx}", f";{colx};", bins[0], bins[1], bins[2])
    if name is not None:
        h1.SetName(name)
    if xtitle is not None:
        h1.GetXaxis().SetTitle(xtitle)
    if ytitle is not None:
        h1.GetYaxis().SetTitle(ytitle)
        
    if df.empty:
        h1.Write()
        return
    for x in df[colx]:
        h1.Fill(x)
    h1.Write()

# ****************************************
def plot_hist_diff(df, colx1, colx2, bins, file, name = None, xtitle = None, ytitle = None):
    file.cd()
    h1 = ROOT.TH1F(f"h_diff_{colx1}_{colx2}", f";{colx1} - {colx2};", bins[0], bins[1], bins[2])
    if name is not None:
        h1.SetName(name)
    if xtitle is not None:
        h1.GetXaxis().SetTitle(xtitle)
    if ytitle is not None:
        h1.GetYaxis().SetTitle(ytitle)

    if df.empty:
        h1.Write()
        return
    for x1, x2 in zip(df[colx1], df[colx2]):
        h1.Fill(x1 - x2)
    h1.Write()

# ****************************************
def plot_2d_scatter(df, colx, coly, bins, file, name = None, xtitle = None, ytitle = None):
    file.cd()
    h1 = ROOT.TH2F(f"h_{colx}_{coly}", f";{colx};{coly}", bins[0][0], bins[0][1], bins[0][2], bins[1][0], bins[1][1], bins[1][2])
    if name is not None:
        h1.SetName(name)
    if xtitle is not None:
        h1.GetXaxis().SetTitle(xtitle)
    if ytitle is not None:
        h1.GetYaxis().SetTitle(ytitle)

    if df.empty:
        h1.Write()
        return
    for x, y in zip(df[colx], df[coly]):
        h1.Fill(x, y)
    h1.Write()
