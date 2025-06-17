#!/usr/bin/env python

import numpy as np
import pandas as pd
from array import array
import matplotlib.pyplot as plt
import ROOT
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
import math
from copy import deepcopy
import utils
import para

# ****************************************
def getConfigPath(dataType):
    if "PbPb" in dataType:
        path = '../config/PbPbConfig.yaml'
    else:    
        path = '../config/ppConfig.yaml'
    return path

# ****************************************
def getDataAnaResultsPath(dataType):
    if dataType == "23Thin":
        path = '../data/LHC23_Thin/LHC23_pass4_Thin_AnalysisResults.root'
    elif dataType == "23PbPb":
        path = '../data/LHC23_PbPb/LHC23PbPb_pass4_AnalysisResults.root'
    elif dataType == "24skimmed":
        path = '../data/LHC24_skimmed/Ana_all.root'
    # elif dataType == "24skimmed_reduced":
    else:
        raise ValueError("Wrong dataType")

    return path

# ****************************************
def getDataAO2DPath(dataType, period=None):
    if dataType == "23Thin":
        path = '../data/LHC23_Thin/LHC23_pass4_Thin_AO2D.root'
    elif dataType == "23PbPb":
        path = '../data/LHC23_PbPb/LHC23PbPb_pass4_AO2D.root'
    elif dataType == "24skimmed":
        path = ['../data/LHC24_skimmed/LHC24am_pass1_skimmed_AO2D.root', '../data/LHC24_skimmed/LHC24an_pass1_skimmed_AO2D.root', '../data/LHC24_skimmed/LHC24ao_pass1_skimmed_AO2D.root']
    elif dataType ==  "24skimmed_reduced":
        path = ['../data/backup_newReduced/LHC24amanao_pass1_skimmed_reduced_AO2D.root']
    elif dataType == "24skimmed_newReduced":
        path = '../data/newReduced/LHC24amanao_newRedcued_AO2D.root'
    elif dataType == "24newSkimmed":
        path = []
        if "am" in period:
            path.append('../data/newSkimmed/LHC24am_pass1_skimmed_AO2D.root')
        if "an" in period:
            path.append('../data/newSkimmed/LHC24an_pass1_skimmed_AO2D.root')
        if "ao" in period:
            path.append('../data/newSkimmed/LHC24ao_pass1_skimmed_AO2D.root')
    else:
        raise ValueError("Wrong dataType")

    return path

# ****************************************
def getMCAO2DPath(dataType, period=None):
    if dataType == "23Thin":
        path = '../data/LHC23_Thin/MCLHC24b2b_AO2D.root'
    elif dataType == "23PbPb":
        path = '../data/LHC23_PbPb/MCLHC24i5_AO2D.root'
    elif dataType == "24skimmed" or dataType == "24skimmed_reduced" or dataType == "24skimmed_newReduced":
        if period == "25a3":
            path = '../data/LHC24_skimmed/MCLHC25a3_AO2D.root'
        else:
            path = ['../data/LHC23_Thin/MCLHC24b2b_AO2D.root', '../data/LHC23_Thin/MCLHC24b2c_AO2D.root', '../data/LHC24_skimmed/MCLHC25a3_AO2D.root']
    else:
        raise ValueError("Wrong dataType")

    return path

# ****************************************
def getBkgAO2DPath(dataType, method, period=None):
    if method == "LikeSign":
        if dataType == "23Thin":
            path = '../data/LHC23_Thin/LHC23_pass4_Thin_LikeSign_AO2D.root'
        elif dataType == "23PbPb":
            path = '../data/LHC23_PbPb/LHC23PbPb_pass4_LikeSign_AO2D.root'
        elif dataType == "24skimmed" or dataType == "24newSkimmed" or dataType == "24skimmed_reduced" or dataType == "24skimmed_newReduced":
            path = []
            if "am" in period:
                path.append('../data/newReduced/LHC24am_newReduced_LikeSign_AO2D.root')
            if "an" in period:
                path.append('../data/newReduced/LHC24an_newReduced_LikeSign_AO2D.root')
            if "ao" in period:
                path.append('../data/newReduced/LHC24ao_newReduced_LikeSign_AO2D.root')
        else:
            raise ValueError("No corresponding Like-sign background data")
    else:
        raise ValueError("Wrong Method to get background")

    return path

# ****************************************
def getDataTH(dataType, period=None):
    DataTH = TreeHandler(getDataAO2DPath(dataType, period=period),'O2hyp3bodycands', folder_name='DF*')
    return DataTH

# ****************************************
def getMCTH(dataType, period=None):
    MCTH = TreeHandler(getMCAO2DPath(dataType, period),'O2mchyp3bodycands', folder_name='DF*')
    return MCTH

# ****************************************
def getBkgTH(dataType, method, DataTH = None, period = None):
    if method == "mixed_deuteron":
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_pass1_skimmed_reduced_EM5000000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_newReducedTest_OnlyCutOnH3LDCA_EM5000000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_newReducedTest_noV0CosPAXYCut_EM2000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_newReduced_NoV0Cut_EM5000000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler(["../data/newReduced/LHC24amao_newReduced_mixingDeuteron_AO2D.root", "../data/newReduced/LHC24an_newReduced_mixingDeuteron_AO2D.root"],'O2hyp3bodycands', folder_name='DF*')
        BkgTH = TreeHandler(["../data/EMReduced/LHC24amao_newReduced_mixingDeuteron_AO2D.root", "../data/EMReduced/LHC24an_newReduced_mixingDeuteron_AO2D.root"],'O2hyp3bodycands', folder_name='DF*')
    elif method == "mixed_deuteron_newBin":
        BkgTH = TreeHandler(["../data/EMNewBin/LHC24amao_newReduced_mixingDeuteron_AO2D.root", "../data/EMNewBin/LHC24an_newReduced_mixingDeuteron_AO2D.root"],'O2hyp3bodycands', folder_name='DF*')
    elif method == "mixed_uncorrelated":
        # BkgTH = TreeHandler(["../data/newReduced/LHC24amao_newReduced_mixingProton_AO2D.root", "../data/newReduced/LHC24an_newReduced_mixingProton_AO2D.root"],'O2hyp3bodycands', folder_name='DF*')
        BkgTH = TreeHandler(["../data/EMReduced/LHC24amao_newReduced_mixingProton_AO2D.root", "../data/EMReduced/LHC24an_newReduced_mixingProton_AO2D.root",
                             "../data/EMReduced/LHC24amao_newReduced_mixingPion_AO2D.root", "../data/EMReduced/LHC24an_newReduced_mixingPion_AO2D.root"],'O2hyp3bodycands', folder_name='DF*')
    elif method == "Sideband":
        if DataTH == None:
            raise ValueError("Input dataTH for background tree")
        else:
            BkgTH = DataTH.get_subset('fM < 2.98 or fM > 3.005')
    elif method == "EM" or method == "LikeSign":                
        BkgTH = TreeHandler(getBkgAO2DPath(dataType, method, period),'O2hyp3bodycands', folder_name='DF*')
    else:
        raise ValueError("Wrong Method to get background")

    return BkgTH

# ****************************************
def getEventNumber(dataType, period = None):
    if "skimmed" in dataType.lower():
        # anaQA = ROOT.TFile(getDataAnaResultsPath(dataType), "READ")
        # dir = anaQA.Get("threebody-reco-task")
        # zorroSum = dir.Get("zorroSummary;1")
        # return zorroSum.getNormalisationFactor(0)
        if dataType == "24skimmed":
            return 1.2403e+12
        elif dataType == "24newSkimmed":
            return 1.1528149e+12
        elif dataType == "24skimmed_newReduced":
            return 1.1804801e+12
        elif dataType == "24skimmed_reduced":
            return 1.2977420e+12
        raise ValueError("Wrong dataType")
    else:
        anaQA = ROOT.TFile(getDataAnaResultsPath(dataType), "READ")
        hEventCounter = anaQA.Get("threebody-reco-task/hEventCounter")
        return hEventCounter.GetBinContent(3)

# ****************************************
def getHypertritonPtShape(dataType):
    if "PbPb" in dataType:
        print("Not implemented yet")
    else:
        f = ROOT.TFile("../CC_file/pt_analysis_antimat_2024_newcut.root", "READ")
    tf = f.Get("std/mtexpo")
    return tf

# ****************************************
def getAbsorpFactor(CENT_BIN_LIST, PT_BIN_LIST, absorp_file = "../CC_file/absorption_histos_3b.root"):
    if PT_BIN_LIST != [[[2, 3], [3, 5]]]:
        raise ValueError("Wrong PT_BIN")
    fAbsorption = ROOT.TFile(absorp_file, "READ")
    hAbsorptionMatter = fAbsorption.Get("x1.5/h_abso_frac_pt_mat")
    hAbsorptionAntiMatter = fAbsorption.Get("x1.5/h_abso_frac_pt_antimat")
    AbsorbFactor = utils.createEmptyList( [len(CENT_BIN_LIST)] )
    for icent, centbin in enumerate(CENT_BIN_LIST):
        for ipt, ptbin in enumerate( PT_BIN_LIST[icent] ):
            fabso = hAbsorptionMatter.GetBinContent(ipt + 1)
            fabso += hAbsorptionAntiMatter.GetBinContent(ipt + 1)
            fabso = fabso / 2
            AbsorbFactor[icent].append(fabso)
    return AbsorbFactor

# ****************************************
# def getAbsorpSyst(CENT_BIN_LIST, PT_BIN_LIST, absorp_syst_file = "../absorb/AbsorpResults_1.5.root"):
#     ''' Absorption results with larger cross section used to calculate systematical uncertainties '''

#     fAbsorption = ROOT.TFile(absorp_syst_file, "READ")
#     #hAbsorption = [fAbsorption.Get("AbsorbRatio_0_10"), fAbsorption.Get("AbsorbRatio_10_30"), fAbsorption.Get("AbsorbRatio30_50"), fAbsorption.Get("AbsorbRatio50_90")]
#     hAbsorption = []
#     AbsorbFactor = utils.createEmptyList( [len(CENT_BIN_LIST)] )
#     for icent, centbin in enumerate(CENT_BIN_LIST):
#         hAbsorption.append( fAbsorption.Get("AbsorpRatio_" + str(centbin[0]) + "_" + str(centbin[1]) ) )
#         for ipt, ptbin in enumerate( PT_BIN_LIST[icent] ):
#             AbsorbFactor[icent].append(hAbsorption[icent].GetBinContent( hAbsorption[icent].FindBin( (ptbin[0] + ptbin[1])/2. )) )
#     return AbsorbFactor

# ****************************************
def getH3L2bodyYield(PT_BIN_LIST):
    # Readin hypertriton yield from 2-body decay analysis
    f = ROOT.TFile("../CC_file/spectra_inel.root", "READ")
    if PT_BIN_LIST == [[[2, 2.3], [2.3, 2.6], [2.6, 5]]]:
        index_bins = [[3, 3], [4, 4], [5, 8]]
    elif PT_BIN_LIST == [[[2, 3], [3, 5]]]:
        index_bins = [[3, 5], [6, 8]]
    elif PT_BIN_LIST == [[[2, 2.6], [2.6, 5]]]:
        index_bins = [[3, 4], [5, 8]]
    elif PT_BIN_LIST == [[[2, 5]]]:
        index_bins = [[3, 8]]
    else:
        raise ValueError("Wrong PT_BIN")
    hStat = f.Get("hStat")
    hSystRMS = f.Get("hSystRMS")
    rawyield_2body = []
    statunc_2body = []
    systunc_2body = []
    statE = np.zeros(1)
    systE = np.zeros(1)
    for ibin, histbin in enumerate(index_bins):
        ptbin = PT_BIN_LIST[0][ibin]
        res = hStat.IntegralAndError(histbin[0], histbin[1], statE, "width")
        hSystRMS.IntegralAndError(histbin[0], histbin[1], systE, "width")
        res = res / (ptbin[1] - ptbin[0]) * para.BR_2body
        statE = statE / (ptbin[1] - ptbin[0]) * para.BR_2body
        systE = systE / (ptbin[1] - ptbin[0]) * para.BR_2body
        print(res, statE[0], systE[0])
        rawyield_2body.append(res)
        statunc_2body.append(statE[0])
        systunc_2body.append(systE[0])
    rawyield_2body = np.array(rawyield_2body)
    statunc_2body = np.array(statunc_2body) 
    systunc_2body = np.array(systunc_2body)
    return (rawyield_2body, statunc_2body, systunc_2body)

# ****************************************
def getH3L2bodyYieldHist(PT_BIN_LIST, icent=0):
    (rawyield_2body, statunc_2body, systunc_2body) = getH3L2bodyYield(PT_BIN_LIST)
    pt_bins = np.array([pt[0] for pt in PT_BIN_LIST[icent]] + [PT_BIN_LIST[icent][-1][1]])
    pt_bins = array('d', pt_bins.astype(np.float64))
    h2bodyStat = ROOT.TH1F("h2bodyStat", ";#it{p}_{T} (GeV/c);R", len(pt_bins) - 1, pt_bins)
    h2bodySyst = ROOT.TH1F("h2bodySyst", ";#it{p}_{T} (GeV/c);R", len(pt_bins) - 1, pt_bins)
    for ipt, ptbin in enumerate(PT_BIN_LIST[0]):
        h2bodyStat.SetBinContent(ipt+1, rawyield_2body[ipt])
        h2bodyStat.SetBinError(ipt+1, statunc_2body[ipt])
        h2bodySyst.SetBinContent(ipt+1, rawyield_2body[ipt])
        h2bodySyst.SetBinError(ipt+1, systunc_2body[ipt])
    return (h2bodyStat, h2bodySyst)
