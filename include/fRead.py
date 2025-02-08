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
def getDataAO2DPath(dataType):
    if dataType == "23Thin":
        path = '../data/LHC23_Thin/LHC23_pass4_Thin_AO2D.root'
    elif dataType == "23skimmed":
        path = '../data/LHC23_skimmed/LHC23_pass4_skimmed_AO2D.root'
    elif dataType == "23PbPb":
        path = '../data/LHC23_PbPb/LHC23PbPb_pass4_AO2D.root'
    elif dataType == "24skimmed":
        path = ['../data/LHC24_skimmed/LHC24am_pass1_skimmed_AO2D.root', '../data/LHC24_skimmed/LHC24an_pass1_skimmed_AO2D.root', '../data/LHC24_skimmed/LHC24ao_pass1_skimmed_AO2D.root']
    else:
        raise ValueError("Wrong dataType")

    return path

# ****************************************
def getMCAO2DPath(dataType):
    if dataType == "23Thin" or dataType == "23skimmed":
        path = '../data/LHC23_Thin/MCLHC24b2b_AO2D.root'
    elif dataType == "23PbPb":
        path = '../data/LHC23_PbPb/MCLHC24i5_AO2D.root'
    elif dataType == "24skimmed":
        path = ['../data/LHC23_Thin/MCLHC24b2b_AO2D.root', '../data/LHC23_Thin/MCLHC24b2c_AO2D.root']
    else:
        raise ValueError("Wrong dataType")

    return path

# ****************************************
def getDataAnaResultsPath(dataType):
    if dataType == "23Thin":
        path = '../data/LHC23_Thin/LHC23_pass4_Thin_AnalysisResults.root'
    elif dataType == "23skimmed":
        path = '../data/LHC23_skimmed/LHC23_pass4_skimmed_AnalysisResults.root'
    elif dataType == "23PbPb":
        path = '../data/LHC23_PbPb/LHC23PbPb_pass4_AnalysisResults.root'
    elif dataType == "24skimmed":
        path = '../data/LHC24_skimmed/Ana_all.root'
    else:
        raise ValueError("Wrong dataType")

    return path

# ****************************************
def getDataTH(dataType):
    
    DataTH = TreeHandler(getDataAO2DPath(dataType),'O2hyp3bodycands', folder_name='DF*')
    return DataTH

# ****************************************
def getMCTH(dataType):
    
    MCTH = TreeHandler(getMCAO2DPath(dataType),'O2mchyp3bodycands', folder_name='DF*')
    return MCTH

# ****************************************
def getBkgTH(dataType, method, DataTH = None):

    if method == "Sideband" and DataTH == None:
        raise ValueError("Input dataTH for background tree")

    if method == "Sideband":
        BkgTH = DataTH.get_subset('fM < 2.98 or fM > 3.005')
    elif method == "EM":
        if dataType == "23Thin" or dataType == "23skimmed":
            BkgTH = TreeHandler('../data/LHC23_Thin/LHC23_pass4_Thin_small_EM_AO2D.root','O2hyp3bodycands', folder_name='DF*')
        # elif dataType == "23PbPb":
        # elif dataType == "24skimmed":
        else:
             raise ValueError("No corresponding background data")
    elif method == "LikeSign":
        if dataType == "23Thin" or dataType == "23skimmed":
            BkgTH = TreeHandler('../data/LHC23_skimmed/LHC23_pass4_skimmed_LikeSign_AO2D.root','O2hyp3bodycands', folder_name='DF*')
        elif dataType == "23PbPb":
            BkgTH = TreeHandler('../data/LHC23_PbPb/LHC23PbPb_pass4_LikeSign_AO2D.root','O2hyp3bodycands', folder_name='DF*')
        #elif dataType == "24skimmed":
        else:
            raise ValueError("No corresponding background data")
    else:
        raise ValueError("Wrong Method")

    return BkgTH

# ****************************************
def getEventNumber(dataType):
    if "skimmed" in dataType:
        # anaQA = ROOT.TFile(getDataAnaResultsPath(dataType), "READ")
        # dir = anaQA.Get("threebody-reco-task")
        # zorroSum = dir.Get("zorroSummary;1")
        # return zorroSum.getNormalisationFactor(0)
        if dataType == "24skimmed":
            return 1.2635364e+12
        raise ValueError("skimmed data to be implemented")
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
def getAbsorpFactor(CENT_BIN_LIST, PT_BIN_LIST, absorp_file = "../absorb/AbsorpResults_1.5.root"):
    fAbsorption = ROOT.TFile(absorp_file, "READ")
    #hAbsorption = [fAbsorption.Get("AbsorbRatio_0_10"), fAbsorption.Get("AbsorbRatio_10_30"), fAbsorption.Get("AbsorbRatio30_50"), fAbsorption.Get("AbsorbRatio50_90")]
    hAbsorption = []
    AbsorbFactor = myH.createEmptyList( [len(CENT_BIN_LIST)] )
    for icent, centbin in enumerate(CENT_BIN_LIST):
        hAbsorption.append( fAbsorption.Get("AbsorpRatio_" + str(centbin[0]) + "_" + str(centbin[1]) ) )
        for ipt, ptbin in enumerate( PT_BIN_LIST[icent] ):
            AbsorbFactor[icent].append(hAbsorption[icent].GetBinContent( hAbsorption[icent].FindBin( (ptbin[0] + ptbin[1])/2. )) )
    return AbsorbFactor

# ****************************************
def getAbsorpSyst(CENT_BIN_LIST, PT_BIN_LIST, absorp_syst_file = "../absorb/AbsorpResults_1.5.root"):
    ''' Absorption results with larger cross section used to calculate systematical uncertainties '''

    fAbsorption = ROOT.TFile(absorp_syst_file, "READ")
    #hAbsorption = [fAbsorption.Get("AbsorbRatio_0_10"), fAbsorption.Get("AbsorbRatio_10_30"), fAbsorption.Get("AbsorbRatio30_50"), fAbsorption.Get("AbsorbRatio50_90")]
    hAbsorption = []
    AbsorbFactor = myH.createEmptyList( [len(CENT_BIN_LIST)] )
    for icent, centbin in enumerate(CENT_BIN_LIST):
        hAbsorption.append( fAbsorption.Get("AbsorpRatio_" + str(centbin[0]) + "_" + str(centbin[1]) ) )
        for ipt, ptbin in enumerate( PT_BIN_LIST[icent] ):
            AbsorbFactor[icent].append(hAbsorption[icent].GetBinContent( hAbsorption[icent].FindBin( (ptbin[0] + ptbin[1])/2. )) )
    return AbsorbFactor