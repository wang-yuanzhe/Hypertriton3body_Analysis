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
import para

from load_common import add_common_path
add_common_path()
from constants import hypertriton_mass, deuteron_mass, proton_mass, pion_charged_mass
import utils

# ****************************************
def extend_df(df, isMC=False):
    df.eval('fRad = sqrt((fX * fX) + (fY * fY))', inplace=True)
    df.eval('fPt = sqrt((fPx * fPx) + (fPy * fPy))', inplace=True)
    df.eval('fP = sqrt((fPx * fPx) + (fPy * fPy) + (fPz * fPz))', inplace=True)
    df.eval(f'fE = sqrt((fPx * fPx) + (fPy * fPy) + (fPz * fPz) + ({hypertriton_mass} * {hypertriton_mass}))', inplace=True)
    df.eval('fEta = arccosh(fP/fPt)', inplace=True)
    df.eval('fRap = 0.5 * log((fE + fPz) / (fE - fPz))', inplace=True)
    df.eval('fDecLen = sqrt(fX**2 + fY**2 + fZ**2)', inplace=True)
    df.eval(f'fCt = fDecLen * {hypertriton_mass} / fP', inplace=True)
    df.eval('fPtPr = sqrt((fPxTrackPr * fPxTrackPr) + (fPyTrackPr * fPyTrackPr))', inplace=True)
    df.eval('fPtPi = sqrt((fPxTrackPi * fPxTrackPi) + (fPyTrackPi * fPyTrackPi))', inplace=True)
    df.eval('fPtDe = sqrt((fPxTrackDe * fPxTrackDe) + (fPyTrackDe * fPyTrackDe))', inplace=True)
    df.eval('fPPr = sqrt((fPxTrackPr * fPxTrackPr) + (fPyTrackPr * fPyTrackPr) + (fPzTrackPr * fPzTrackPr))', inplace=True)
    df.eval('fPPi = sqrt((fPxTrackPi * fPxTrackPi) + (fPyTrackPi * fPyTrackPi) + (fPzTrackPi * fPzTrackPi))', inplace=True)
    df.eval('fPDe = sqrt((fPxTrackDe * fPxTrackDe) + (fPyTrackDe * fPyTrackDe) + (fPzTrackDe * fPzTrackDe))', inplace=True)
    df.eval('fEtaPr = arccosh(fPPr/fPtPr)', inplace=True)
    df.eval('fEtaPi = arccosh(fPPi/fPtPi)', inplace=True)
    df.eval('fEtaDe = arccosh(fPDe/fPtDe)', inplace=True)
    df.eval(f'fM2PrPi = (sqrt(fPxTrackPr*fPxTrackPr + fPyTrackPr*fPyTrackPr + fPzTrackPr*fPzTrackPr + {proton_mass}*{proton_mass}) + sqrt(fPxTrackPi*fPxTrackPi + fPyTrackPi*fPyTrackPi + fPzTrackPi*fPzTrackPi + {pion_charged_mass}*{pion_charged_mass})) * (sqrt(fPxTrackPr*fPxTrackPr + fPyTrackPr*fPyTrackPr + fPzTrackPr*fPzTrackPr + {proton_mass}*{proton_mass}) + sqrt(fPxTrackPi*fPxTrackPi + fPyTrackPi*fPyTrackPi + fPzTrackPi*fPzTrackPi + {pion_charged_mass}*{pion_charged_mass})) - ((fPxTrackPr + fPxTrackPi) * (fPxTrackPr + fPxTrackPi) + (fPyTrackPr + fPyTrackPi) * (fPyTrackPr + fPyTrackPi) + (fPzTrackPr + fPzTrackPi) * (fPzTrackPr + fPzTrackPi))', inplace=True)
    df.eval('fMPrPi = sqrt(fM2PrPi)', inplace=True)
    df.eval(f'fM2PiDe = (sqrt(fPxTrackPi*fPxTrackPi + fPyTrackPi*fPyTrackPi + fPzTrackPi*fPzTrackPi + {pion_charged_mass}*{pion_charged_mass}) + sqrt(fPxTrackDe*fPxTrackDe + fPyTrackDe*fPyTrackDe + fPzTrackDe*fPzTrackDe + 1.875613*1.875613)) * (sqrt(fPxTrackPi*fPxTrackPi + fPyTrackPi*fPyTrackPi + fPzTrackPi*fPzTrackPi + {pion_charged_mass}*{pion_charged_mass}) + sqrt(fPxTrackDe*fPxTrackDe + fPyTrackDe*fPyTrackDe + fPzTrackDe*fPzTrackDe + 1.875613*1.875613)) - ((fPxTrackPi + fPxTrackDe) * (fPxTrackPi + fPxTrackDe) + (fPyTrackPi + fPyTrackDe) * (fPyTrackPi + fPyTrackDe) + (fPzTrackPi + fPzTrackDe) * (fPzTrackPi + fPzTrackDe))', inplace=True)
    df.eval('fDCAXYsumDaughtersToPV = fDCAXYTrackPrToPV + fDCAXYTrackPiToPV + fDCAXYTrackDeToPV', inplace=True)
    df.eval('fDCAvtxQuadSumAv = (fDCATrackPrToSV**2 + fDCATrackPiToSV**2 + fDCATrackDeToSV**2) / 3', inplace=True)
    df.eval('fDCAvtxDaughtersSum = fDCATrackPrToSV + fDCATrackPiToSV + fDCATrackDeToSV', inplace=True)
    # df.eval('fDCATrackPrToPV = sqrt(fDCAXYTrackPrToPV*fDCAXYTrackPrToPV + fDCAZTrackPrToPV*fDCAZTrackPrToPV)', inplace=True)
    # df.eval('fDCATrackPiToPV = sqrt(fDCAXYTrackPiToPV*fDCAXYTrackPiToPV + fDCAZTrackPiToPV*fDCAZTrackPiToPV)', inplace=True)
    # df.eval('fDCATrackDeToPV = sqrt(fDCAXYTrackDeToPV*fDCAXYTrackDeToPV + fDCAZTrackDeToPV*fDCAZTrackDeToPV)', inplace=True)
    if isMC:
        df.eval('fGenRad = sqrt((fGenX * fGenX) + (fGenY * fGenY))', inplace=True)
        df.eval('fGenPt = sqrt((fGenPx * fGenPx) + (fGenPy * fGenPy))', inplace=True)
        df.eval('fResoPt = fPt - fGenPt', inplace=True)
        # df.eval('pullPt = (pT - genPt)/fPtErr', inplace=True)
        df.eval('fResoX = fX - fGenX', inplace=True)
        df.eval('fResoY = fY - fGenY', inplace=True)
        df.eval('fResoZ = fZ - fGenZ', inplace=True)
        # df.eval('fPullX = (fX - fGenX)/fVtxCovMat[0]', inplace=True)
        # df.eval('fPullY = (fY - fGenY)/fVtxCovMat[2]', inplace=True)
        # df.eval('fPullZ = (fZ - fGenZ)/fVtxCovMat[5]', inplace=True)
    
    df.drop(["fPxTrackPr", "fPyTrackPr", "fPzTrackPr", "fPxTrackPi", "fPyTrackPi", "fPzTrackPi", "fPxTrackDe", "fPyTrackDe", "fPzTrackDe"], axis=1, inplace=True)

# ****************************************
def getConfigPath(dataType):
    if "PbPb" in dataType:
        path = '../config/PbPbConfig.yaml'
    else:    
        path = '../config/ppConfig.yaml'
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
def getBkgPD(dataType, method, DataPD = None, col_mass="fMass"):
    if method == "mixed_deuteron":
        # BkgPD = utils.getDF_fromFile("../data/newReduced/LHC24amao_pass1_skimmed_reduced_EM5000000_AO2D.root", 'O2hyp3bodycands', folder_name='DF*')
        # BkgPD = utils.getDF_fromFile("../data/newReduced/LHC24amao_newReducedTest_OnlyCutOnH3LDCA_EM5000000_AO2D.root", 'O2hyp3bodycands', folder_name='DF*')
        # BkgPD = utils.getDF_fromFile("../data/newReduced/LHC24amao_newReducedTest_noV0CosPAXYCut_EM2000_AO2D.root", 'O2hyp3bodycands', folder_name='DF*')
        # BkgPD = utils.getDF_fromFile("../data/newReduced/LHC24amao_newReduced_NoV0Cut_EM5000000_AO2D.root", 'O2hyp3bodycands', folder_name='DF*')
        # BkgPD = utils.getDF_fromFile(["../data/newReduced/LHC24amao_newReduced_mixingDeuteron_AO2D.root", "../data/newReduced/LHC24an_newReduced_mixingDeuteron_AO2D.root"], 'O2hyp3bodycands', folder_name='DF*')
        BkgPD = utils.getDF_fromFile(["../data/before25QM/EMReduced/LHC24amao_newReduced_mixingDeuteron_AO2D.root", "../data/before25QM/EMReduced/LHC24an_newReduced_mixingDeuteron_AO2D.root"], 'O2hyp3bodycands', folder_name='DF*')
    elif method == "mixed_deuteron_newBin":
        BkgPD = utils.getDF_fromFile(["../data/before25QM/EMNewBin/LHC24amao_newReduced_mixingDeuteron_AO2D.root", "../data/before25QM/EMNewBin/LHC24an_newReduced_mixingDeuteron_AO2D.root"], 'O2hyp3bodycands', folder_name='DF*')
    elif method == "mixed_uncorrelated":
        # BkgPD = utils.getDF_fromFile(["../data/newReduced/LHC24amao_newReduced_mixingProton_AO2D.root", "../data/newReduced/LHC24an_newReduced_mixingProton_AO2D.root"], 'O2hyp3bodycands', folder_name='DF*')
        BkgPD = utils.getDF_fromFile(["../data/before25QM/EMReduced/LHC24amao_newReduced_mixingProton_AO2D.root", "../data/before25QM/EMReduced/LHC24an_newReduced_mixingProton_AO2D.root",
                             "../data/before25QM/EMReduced/LHC24amao_newReduced_mixingPion_AO2D.root", "../data/before25QM/EMReduced/LHC24an_newReduced_mixingPion_AO2D.root"], 'O2hyp3bodycands', folder_name='DF*')
    elif method == "Sideband":
        if DataPD == None:
            raise ValueError("Input dataTH for background tree")
        else:
            BkgPD = DataPD.query(f'{col_mass} < 2.98 or {col_mass} > 3.005')
    else:
        raise ValueError("Wrong Method to get background")

    return BkgPD

# ****************************************
def getEventNumber(dataType, AnaFilePath, period = None):
    if "skimmed" in dataType.lower(): # Only work in O2Physics Environment
        # anaQA = ROOT.TFile(AnaFilePath, "READ")
        # dir = anaQA.Get("threebody-reco-task")
        # zorroSum = dir.Get("zorroSummary;1")
        # return zorroSum.getNormalisationFactor(0)
        # if dataType == "24skimmed":
        #     return 1.2403e+12
        # elif dataType == "24newSkimmed":
        #     return 1.1528149e+12
        # elif dataType == "24skimmed_newReduced":
        #     return 1.1804801e+12
        # elif dataType == "24skimmed_reduced":
        #     return 1.2977420e+12
        raise ValueError("Wrong dataType")
    else:
        anaQA = ROOT.TFile(AnaFilePath, "READ")
        hEventCounter = anaQA.Get("decay3body-builder/Counters/hEventCounter")
        return hEventCounter.GetBinContent(2)

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

# ****************************************
def calVirtualLambdaInvM(proton_pt, proton_eta, proton_phi, pion_pt, pion_eta, pion_phi):
    proton_px = proton_pt * np.cos(proton_phi)
    proton_py = proton_pt * np.sin(proton_phi)
    proton_pz = proton_pt * np.sinh(proton_eta)

    pion_px = pion_pt * np.cos(pion_phi)
    pion_py = pion_pt * np.sin(pion_phi)
    pion_pz = pion_pt * np.sinh(pion_eta)

    proton_e = np.sqrt(proton_px**2 + proton_py**2 + proton_pz**2 + proton_mass**2)
    pion_e = np.sqrt(pion_px**2 + pion_py**2 + pion_pz**2 + pion_charged_mass**2)

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

    proton_e = np.sqrt(proton_px**2 + proton_py**2 + proton_pz**2 + proton_mass**2)
    deuteron_e = np.sqrt(deuteron_px**2 + deuteron_py**2 + deuteron_pz**2 + deuteron_mass**2)

    total_e = proton_e + deuteron_e
    total_px = proton_px + deuteron_px
    total_py = proton_py + deuteron_py
    total_pz = proton_pz + deuteron_pz

    invariant_mass = np.sqrt(total_e**2 - total_px**2 - total_py**2 - total_pz**2)
    return invariant_mass

# ****************************************
def calNewElements(df, isNewDataModel = True, isMC = False):
    # df['fDiffRDaughter'] = df['fRadiusBachelor'] - df[['fRadiusProton', 'fRadiusPion']].min(axis=1)
    # df['fDiffRProton'] = df['fRadiusProton'] - df['fVtxRadius']
    # df['fDiffRPion'] = df['fRadiusPion'] - df['fVtxRadius']
    # df['fDiffRBachelor'] = df['fRadiusBachelor'] - df['fVtxRadius']
    ######### temporary fix #########
    if isNewDataModel:
        df['fPtProton'] = np.sqrt(df['fPxTrackPr']**2 + df['fPyTrackPr']**2)
        df['fEtaProton'] = np.arcsinh(df['fPzTrackPr'] / df['fPtProton'])
        df['fPhiProton'] = np.arctan2(df['fPyTrackPr'], df['fPxTrackPr'])
        df['fPtPion'] = np.sqrt(df['fPxTrackPi']**2 + df['fPyTrackPi']**2)
        df['fEtaPion'] = np.arcsinh(df['fPzTrackPi'] / df['fPtPion'])
        df['fPhiPion'] = np.arctan2(df['fPyTrackPi'], df['fPxTrackPi'])
        df['fPtBachelor'] = np.sqrt(df['fPxTrackDe']**2 + df['fPyTrackDe']**2)
        df['fEtaBachelor'] = np.arcsinh(df['fPzTrackDe'] / df['fPtBachelor'])
        df['fPhiBachelor'] = np.arctan2(df['fPyTrackDe'], df['fPxTrackDe'])

        df['fCentrality'] = 1
        df['fPt'] = np.sqrt(df['fPx']**2 + df['fPy']**2)
        df['fTPCNSigmaProton'] = df['fTPCNSigmaPr']
        df['fTPCNSigmaPion'] = df['fTPCNSigmaPi']
        df['fTPCNSigmaBachelor'] = df['fTPCNSigmaDe']
        df['fTOFNSigmaBachelor'] = df['fTOFNSigmaDe']
        df['fDCADaughters'] = df['fDCAVtxToDaughtersAv']
        df['fM'] = df['fMass']
        if 'fDCAZTrackPiToPV' in df.columns:
            df['fDCAPionToPV'] = np.sqrt(df['fDCAXYTrackPiToPV']**2 + df['fDCAZTrackPiToPV']**2)
        else:
            df['fDCAPionToPV'] = df['fDCATrackPiToPV']
        if isMC:
            df['fGenPt'] = np.sqrt(df['fGenPx']**2 + df['fGenPy']**2)
            df['fSurvivedEventSelection'] = df['fIsSurvEvSel']
            df['fIsSignal'] = df['fIsTrueH3L'] | df['fIsTrueAntiH3L']
            df['fGenRapidity'] = df['fGenRap']
    #################################
    df['fMVirtualLambda'] = calVirtualLambdaInvM(df['fPtProton'], df['fEtaProton'], df['fPhiProton'], df['fPtPion'], df['fEtaPion'], df['fPhiPion'])
    df['fMdp'] = calMdp(df['fPtProton'], df['fEtaProton'], df['fPhiProton'], df['fPtBachelor'], df['fEtaBachelor'], df['fPhiBachelor'])

# ****************************************
def fix_oldDataFrame(df):
    df.eval('fMass = fM', inplace=True)
    df.eval('fPDe = fPtBachelor * cosh(fEtaBachelor)', inplace=True)
    df.eval(f'fE = sqrt(fP * fP + {hypertriton_mass} * {hypertriton_mass})', inplace=True)
    df.eval('fPz = fPtProton*sinh(fEtaProton) + fPtPion*sinh(fEtaPion) + fPtBachelor*sinh(fEtaBachelor)', inplace=True)
    df.eval('fRapidity = 0.5 * log((fE + fPz) / (fE - fPz))', inplace=True)