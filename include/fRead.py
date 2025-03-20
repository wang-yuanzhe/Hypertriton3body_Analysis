#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ROOT
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
import math
from copy import deepcopy
import myHeader as myH

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
    elif dataType == "24skimmed_reduced":
        path = []
        if "am" in period:
            path.append('../data/reducedData/LHC24am_pass1_skimmed_reduced_AO2D.root')
        if "an" in period:
            path.append('../data/reducedData/LHC24an_pass1_skimmed_reduced_AO2D.root')
        if "ao" in period:
            path.append('../data/reducedData/LHC24ao_pass1_skimmed_reduced_AO2D.root')
    else:
        raise ValueError("Wrong dataType")

    return path

# ****************************************
def getMCAO2DPath(dataType, period=None):
    if dataType == "23Thin":
        path = '../data/LHC23_Thin/MCLHC24b2b_AO2D.root'
    elif dataType == "23PbPb":
        path = '../data/LHC23_PbPb/MCLHC24i5_AO2D.root'
    elif dataType == "24skimmed" or dataType == "24skimmed_reduced":
        if period == "25a3":
            path = '../data/LHC24_skimmed/MCLHC25a3_AO2D.root'
        else:
            path = ['../data/LHC23_Thin/MCLHC24b2b_AO2D.root', '../data/LHC23_Thin/MCLHC24b2c_AO2D.root', '../data/LHC24_skimmed/MCLHC25a3_AO2D.root']
    else:
        raise ValueError("Wrong dataType")

    return path

# ****************************************
def getBkgAO2DPath(dataType, method, period=None):
    if method == "EM":
        if dataType == "23Thin":
            path = '../data/LHC23_Thin/LHC23_pass4_Thin_small_EM_AO2D.root'
        elif dataType == "24skimmed_reduced":
            if period == "24am":
                path = '../data/reducedData/LHC24am_pass1_skimmed_reduced_EM5000000_AO2D.root'
            elif period == "24amao":
                path = '../data/reducedData/LHC24amao_pass1_skimmed_reduced_EM5000000_AO2D.root'
            else:
                raise ValueError("No corresponding EM background data")
        elif dataType == "24skimmed_newReduced":
            if period == "24amao":
                path = "../data/newReduced/LHC24amao_pass1_skimmed_reduced_EM5000000_AO2D.root"
            else:
                raise ValueError("No corresponding EM background data")
        else:
             raise ValueError("No corresponding EM background data")
    elif method == "LikeSign":
        if dataType == "23Thin":
            path = '../data/LHC23_Thin/LHC23_pass4_Thin_LikeSign_AO2D.root'
        elif dataType == "23PbPb":
            path = '../data/LHC23_PbPb/LHC23PbPb_pass4_LikeSign_AO2D.root'
        elif dataType == "24skimmed":
            path = '../data/LHC23_Thin/LHC23_pass4_Thin_LikeSign_AO2D.root'
        elif dataType == "24skimmed_reduced":
            path = '../data/reducedData/LHC24am_pass1_skimmed_reduced_LikeSign_AO2D.root'
        elif dataType == "24skimmed_newReduced":
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
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_pass1_skimmed_newReducedTest_EM5000000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_newReducedTest_NoCutOnCosPAV0_EM2000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_newReducedTest_OnlyCutOnH3LDCA_EM5000000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_newReducedTest_noV0CosPAXYCut_EM2000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_newReduced_NoV0Cut_EM5000000_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler("../data/newReduced/LHC24amao_pass1_skimmed_reduced_EM5000000_pdmixed_AO2D.root",'O2hyp3bodycands', folder_name='DF*')
        # BkgTH = TreeHandler(["../data/newReduced/LHC24amao_merged_newReducedTest_posZDiff2.5_EM5000000_AO2D.root", "../data/newReduced/LHC24an_merged_newReducedTest_posZDiff2.5_EM5000000_AO2D.root"],'O2hyp3bodycands', folder_name='DF*')
        BkgTH = TreeHandler(["../data/newReduced/LHC24amao_newReduced_mixingDeuteron_AO2D.root", "../data/newReduced/LHC24an_newReduced_mixingDeuteron_AO2D.root"],'O2hyp3bodycands', folder_name='DF*')
    elif method == "mixed_proton":
        BkgTH = TreeHandler(["../data/newReduced/LHC24amao_newReduced_mixingProton_AO2D.root", "../data/newReduced/LHC24an_newReduced_mixingProton_AO2D.root"],'O2hyp3bodycands', folder_name='DF*')
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
    if "skimmed" in dataType:
        # anaQA = ROOT.TFile(getDataAnaResultsPath(dataType), "READ")
        # dir = anaQA.Get("threebody-reco-task")
        # zorroSum = dir.Get("zorroSummary;1")
        # return zorroSum.getNormalisationFactor(0)
        if dataType == "24skimmed":
            return 1.2403e+12
        elif dataType == "24skimmed_reduced":
            return 1.1804801e+12
            # num = 0
            # if "am" in period:
            #     num += 4.8902132e+11
            # if "an" in period:
            #     num += 
            # if "ao" in period: 
            #     num += 
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
    AbsorbFactor = myH.createEmptyList( [len(CENT_BIN_LIST)] )
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
#     AbsorbFactor = myH.createEmptyList( [len(CENT_BIN_LIST)] )
#     for icent, centbin in enumerate(CENT_BIN_LIST):
#         hAbsorption.append( fAbsorption.Get("AbsorpRatio_" + str(centbin[0]) + "_" + str(centbin[1]) ) )
#         for ipt, ptbin in enumerate( PT_BIN_LIST[icent] ):
#             AbsorbFactor[icent].append(hAbsorption[icent].GetBinContent( hAbsorption[icent].FindBin( (ptbin[0] + ptbin[1])/2. )) )
#     return AbsorbFactor

# ****************************************
def getH3L2bodyYield(PT_BIN_LIST):
    # Readin hypertriton yield from 2-body decay analysis
    if PT_BIN_LIST == [[[2, 2.4], [2.4, 3.5], [3.5, 5]]]: # Temparary solution to PT_BIN [[[2, 2.4], [2.4, 3.5], [3.5, 5]]]
        raise ValueError("Wrong PT_BIN")
        # f = ROOT.TFile("../CC_file/pt_analysis_antimat_2024_newcut.root", "READ")
        # index_bins = [[2, 2], [3, 5], [6, 7]]
    elif PT_BIN_LIST == [[[2, 2.3], [2.3, 2.6], [2.6, 5]]]: # Temparary solution to PT_BIN [[[2, 2.3], [2.3, 2.6], [2.6, 5]]]
        f = ROOT.TFile("../CC_file/spectra_inel.root", "READ")
        index_bins = [[3, 3], [4, 4], [5, 8]]
    elif PT_BIN_LIST == [[[2, 3], [3, 5]]]:
        f = ROOT.TFile("../CC_file/spectra_inel.root", "READ")
        index_bins = [[3, 5], [6, 8]]
    elif PT_BIN_LIST == [[[2, 2.6], [2.6, 5]]]:
        f = ROOT.TFile("../CC_file/spectra_inel.root", "READ")
        index_bins = [[3, 4], [5, 8]]
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
        res = res / (ptbin[1] - ptbin[0]) * 0.25
        statE = statE / (ptbin[1] - ptbin[0]) * 0.25
        systE = systE / (ptbin[1] - ptbin[0]) * 0.25
        print(res, statE[0], systE[0])
        rawyield_2body.append(res)
        statunc_2body.append(statE[0])
        systunc_2body.append(systE[0])
    rawyield_2body = np.array(rawyield_2body)
    statunc_2body = np.array(statunc_2body) 
    systunc_2body = np.array(systunc_2body)
    return (rawyield_2body, statunc_2body, systunc_2body)