#!/usr/bin/env python
# coding: utf-8

import numpy as np
import yaml
from hipe4ml.model_handler import ModelHandler
import ROOT
import math
import pickle
import sys
sys.path.append('../include')
import myHeader as myH
import para as para
import fRead as fRead
from Data import analysisData
from copy import deepcopy

#ROOT.Math.IntegratorOneDimOptions.SetDefaultRelTolerance(1.E-16)
#ROOT.Math.IntegratorOneDimOptions.SetDefaultIntegrator("Gauss")
ROOT.Math.IntegratorOneDimOptions.SetDefaultIntegrator("GaussLegendre")
#“1-dim integrators”: “Gauss”, “GaussLegendre”, “Adaptive”, “AdaptiveSingular” “NonAdaptive”
myH.gStyleInit()

# Set paramters
CENT_BIN_LIST = para.CENT_BIN_LIST
PT_BIN_LIST = para.PT_BIN_LIST
MASS_BIN = para.MASS_BIN

sigpdf = ["DSCB"]
bkgpdf = ["Argus"] #use the last one to get the final result

MODEL_Eff_LIST = para.MODEL_Eff_LIST

ifTellMatter = False
dataType = "24skimmed"

savePrefix = "Plot/"
ModelPath = "Model/" + dataType

score_eff_arrays_dict = pickle.load(
    open(ModelPath + "/file_score_eff_dict", "rb"))

config_file_path = fRead.getConfigPath(dataType)
config_file = open(config_file_path, 'r')
config = yaml.full_load(config_file)

# Readin dataset and model
DataTH = fRead.getDataTH(dataType)
MCTH = fRead.getMCTH(dataType)
BkgTH = fRead.getBkgTH(dataType, "Sideband", DataTH)

DataPDRaw = DataTH.get_data_frame()
MCPDRaw = MCTH.get_data_frame()
BkgPDRaw = BkgTH.get_data_frame()

myH.calNewElements(DataPDRaw)
myH.calNewElements(MCPDRaw)
myH.calNewElements(BkgPDRaw)

DataTH.set_data_frame(DataPDRaw)
MCTH.set_data_frame(MCPDRaw)
BkgTH.set_data_frame(BkgPDRaw)

if len(CENT_BIN_LIST) != 1:
    raise ValueError("CENT_BIN_LIST should have only one element") # Now we only consider one centrality bin

model_hdl = myH.createEmptyList( [len(CENT_BIN_LIST)] )
for icent, centbin in enumerate(CENT_BIN_LIST):
    for ptbin in PT_BIN_LIST[icent]:
        modelfile_name = ModelPath + '/Model' + "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
        # modelfile_name = ModelPath + '/ModelTest'
        modelReadin = ModelHandler()
        modelReadin.load_model_handler(modelfile_name)
        model_hdl[icent].append(deepcopy(modelReadin))

print("MC dataset:",len(MCPDRaw))
print("Generated MC hypertirton within |y| < 0.5:", len(MCPDRaw.query("fIsSignal == 1 and abs(fGenRapidity) < 0.5")))

# Reweight pt shape
pt_spectrum = fRead.getHypertritonPtShape(dataType)
myH.apply_pt_rejection(MCPDRaw, [pt_spectrum], [[-100, 999]], PT_BIN_LIST) # centrality unavailable
MCPDRaw = MCPDRaw.query("rej == False and fSurvivedEventSelection == True")

print("After pT reweight")
print("MC dataset:",len(MCPDRaw))
print("Generated MC hypertirton within |y| < 0.5:", len(MCPDRaw.query("fIsSignal == 1 and abs(fGenRapidity) < 0.5")))
print("Reconstructed MC hypertirton within 2 <= pT < 5:", len(MCPDRaw.query("fIsSignal == 1 and abs(2 <= fPt < 5)")))

# Produce pd in different intervals with model predictions
# add "output_margin=False" will set the output score as probability

preCut = myH.convert_sel_to_string(config['MLPreSelection'])
print(preCut)

data = analysisData(DataPDRaw, preCut, CENT_BIN_LIST, PT_BIN_LIST, MASS_BIN)
data.addModelPrediction(model_hdl)
mcdata = analysisData(MCPDRaw, preCut, CENT_BIN_LIST, PT_BIN_LIST, MASS_BIN) # cut for pt intervals include fPt > 0
mcdata.addModelPrediction(model_hdl)

# Fit invariant mass distribution to extract signal for both MC and data samples
MCFitpara = myH.createEmptyList( [len(CENT_BIN_LIST), len(PT_BIN_LIST[0])] ) # only for DSCB
with ROOT.TFile(savePrefix + "MCfit.root", "recreate") as outfile:
    for icent, centbin in enumerate(CENT_BIN_LIST):
        for ipt, ptbin in enumerate(PT_BIN_LIST[icent]):
            for isig, sigfunc in enumerate(sigpdf):
                (_, __, paras) = mcdata.invMFit(icent, ipt, sigfunc, bkgpdf="none", isMC=True, ifDrawStats=False, outfile = outfile)
                MCFitpara[icent][ipt].append(paras)

signalCount = myH.createEmptyList( [len(CENT_BIN_LIST), len(PT_BIN_LIST[0]), len(sigpdf), len(bkgpdf)] )
expBkgCount = myH.createEmptyList( [len(CENT_BIN_LIST), len(PT_BIN_LIST[0]), len(sigpdf), len(bkgpdf)] )
signalError = myH.createEmptyList( [len(CENT_BIN_LIST), len(PT_BIN_LIST[0]), len(sigpdf), len(bkgpdf)] )
with ROOT.TFile(savePrefix + "fit.root", "recreate") as outfile:
    for icent, centbin in enumerate(CENT_BIN_LIST):
        savecentdir = outfile.mkdir("Cent"+str(centbin[0]) + "_" + str(centbin[1]))
        for ipt, ptbin in enumerate(PT_BIN_LIST[icent]):
            saveptdir = savecentdir.mkdir("pT"+str(ptbin[0]) + "_" + str(ptbin[1]))
            for isig, sigfunc in enumerate(sigpdf):
                savesigdir = saveptdir.mkdir(sigfunc)
                for ibkg, bkgfunc in enumerate(bkgpdf):
                    savebkgdir = savesigdir.mkdir(bkgfunc)
                    for imodel, modelEff in enumerate(MODEL_Eff_LIST):
                        binkey =  "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
                        model_threshold = score_eff_arrays_dict[binkey][0][imodel]
                        binInfo = " pT " + str(ptbin[0]) + "-" + str(ptbin[1]) + "GeV/c " + " BDTEff=" +str(round(modelEff,2)) + " " + dataType + " " + sigfunc + " " + bkgfunc
                        (signalN, bkgN, paras) = data.invMFit(icent, ipt, sigfunc, bkgfunc, model_threshold, name=binInfo, para=MCFitpara[icent][ipt][isig], ifDebug=True, outfile = savebkgdir)
                        if bkgN < 0:
                            bkgN = 0
                        signalCount[icent][ipt][isig][ibkg].append(signalN.getVal())
                        expBkgCount[icent][ipt][isig][ibkg].append(bkgN)
                        signalError[icent][ipt][isig][ibkg].append(signalN.getError())

# Obtain Absorption factor
# AbsorbFactor = myH.ReadinAbsorp(CENT_BIN_LIST, PT_BIN_LIST)
AbsorbFactor = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] # To be implemented

# Calculate efficiency and expected significance
MCGenSignalCount = myH.createEmptyList( [len(CENT_BIN_LIST)] )
MCRecSignalCount = myH.createEmptyList( [len(CENT_BIN_LIST)] ) #reconstructed hypertriton count
ExpSignalYield = myH.createEmptyList( [len(CENT_BIN_LIST)] ) # expected number of reconstructed hypertriton via 3-body decay
Efficiency = myH.createEmptyList( [len(CENT_BIN_LIST)] )
ExpSignificance = myH.createEmptyList( [len(CENT_BIN_LIST), len(PT_BIN_LIST[0]), len(sigpdf), len(bkgpdf)] )


with ROOT.TFile(savePrefix + "fit.root", "update") as outfile:
    for icent, centbin in enumerate(CENT_BIN_LIST):
        for ipt, ptbin in enumerate(PT_BIN_LIST[icent]):
            binCutMC = f'(fGenPt >= {ptbin[0]}) and (fGenPt < {ptbin[1]})'
            MCGenSignalCount[icent].append(len(MCPDRaw.query(f"fIsSignal == 1 and abs(fGenRapidity) < 0.5 and {binCutMC}")))
            MCRecSignalCount[icent].append(len(mcdata.dataPD[icent][ipt]))
            ExpSignalYield[icent].append( pt_spectrum.Integral(ptbin[0], ptbin[1]) * fRead.getEventNumber(dataType) * 0.4)
            eff = MCRecSignalCount[icent][ipt]/MCGenSignalCount[icent][ipt] #efficiency with total recontructed hypertriton count
            Efficiency[icent].append(eff)
            print("Efficiency in ", f'( {centbin[0]} - {centbin[1]} ) and ( {ptbin[0]} <= pT < {ptbin[1]} ), :', eff)
            print("Expected Yield", pt_spectrum.Integral(ptbin[0], ptbin[1]) / (ptbin[1] - ptbin[0]) )
            print("Exp Raw SignalYield:", ExpSignalYield[icent][ipt] * eff * 2)

            binkey = "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
            # binkey = "ModelTest"
            for isig, sigfunc in enumerate(sigpdf):
                for ibkg, bkgfunc in enumerate(bkgpdf):
                    for imodel, modelEff in enumerate(MODEL_Eff_LIST):
                        bdteff = score_eff_arrays_dict[binkey][1][imodel]
                        corrfactor = eff * bdteff * AbsorbFactor[icent][ipt]
                        expsig = 2 * ExpSignalYield[icent][ipt] * corrfactor
                        if ifTellMatter:
                            expsig = expsig / 2
                        ExpSignificance[icent][ipt][isig][ibkg].append(expsig / math.sqrt(expsig + expBkgCount[icent][ipt][isig][ibkg][imodel]))

        #Plot efficiency versus pt in each centbin
        savecentdir = outfile.Get("Cent"+str(centbin[0]) + "_" + str(centbin[1]))
        c_efficiency = myH.TCanvas("efficiency"+"Cent"+str(centbin[0]) + "_" + str(centbin[1]),'efficiency')
        c_efficiency.cd()
        hMCGenSignalCount = ROOT.TH1F("hMCGenSignalCount", "hMCGenSignalCount", data.hist_ptBins[icent].size - 1, data.hist_ptBins[icent])
        hEff = ROOT.TH1F("hEff", ";#it{p}_{T} (GeV/#it{c});Precut Efficiency * Acceptance", data.hist_ptBins[icent].size - 1, data.hist_ptBins[icent])
        for ipt in range(data.hist_ptBins[icent].size - 1):
            hMCGenSignalCount.SetBinContent(ipt+1, MCGenSignalCount[icent][ipt])
            hMCGenSignalCount.SetBinError(ipt+1, 0)
            hEff.SetBinContent(ipt+1, MCRecSignalCount[icent][ipt])
        hEff.Divide(hMCGenSignalCount)
        hEff.GetYaxis().SetRangeUser(0, 0.08)
        hEff.SetMarkerStyle(8)
        hEff.SetMarkerSize(0.5)
        hEff.Draw("ep")
        c_efficiency.Write()

# QA and systematical uncertainties
SystUnc = myH.createEmptyList( [len(CENT_BIN_LIST)] )
with ROOT.TFile(savePrefix + "fit.root", "update") as outfile:
    for icent, centbin in enumerate(CENT_BIN_LIST):
        SystUnc.append([])
        savecentdir = outfile.Get("Cent"+str(centbin[0]) + "_" + str(centbin[1]))
        for ipt, ptbin in enumerate(PT_BIN_LIST[icent]):
            #BDT efficiency versus output of model
            saveptdir = savecentdir.Get("pT"+str(ptbin[0]) + "_" + str(ptbin[1]))
            #binkey = "Cent" + str(centbin[0]) + "_" + str(centbin[1]) + "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
            binkey = "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
            #binkey = "Test"
            modelOutput_x = score_eff_arrays_dict[binkey][0][0:len(MODEL_Eff_LIST)]
            BDTefficiency_y = score_eff_arrays_dict[binkey][1][0:len(MODEL_Eff_LIST)]
            binInfo = "pT" + str(ptbin[0]) + "-" + str(ptbin[1]) + " GeV/c"
            c_BDTefficiency = myH.TCanvas()
            c_BDTefficiency.cd()
            h_Back_BDTeff = ROOT.TH2F("h_Back_BDTeff"+binInfo, binInfo + ";Model_cut;BDTefficiency", 1,-10,15, 1,0,1.1*np.max(BDTefficiency_y) )
            gr_BDTefficiency = ROOT.TGraph(len(MODEL_Eff_LIST), modelOutput_x, BDTefficiency_y)
            h_Back_BDTeff.Draw()
            gr_BDTefficiency.SetLineColor(4)
            gr_BDTefficiency.Draw("LP same")
            saveptdir.WriteObject(c_BDTefficiency, "BDTefficiency"+binkey)
            CorrectedSignal = [] #list of corrected signal to calculate systematic uncertainties
            for isig, sigfunc in enumerate(sigpdf):
                for ibkg, bkgfunc in enumerate(bkgpdf):
                    binInfo = "pT" + str(ptbin[0]) + "-" + str(ptbin[1]) + " GeV/c" + " " + sigfunc + " " + bkgfunc
                    significance = np.array(ExpSignificance[icent][ipt][isig][ibkg])
                    significanceMulBDTEff = significance * BDTefficiency_y
                    c_ModelSelection = myH.TCanvas()
                    c_ModelSelection.cd()
                    h_Back_modelsel = ROOT.TH2F("h_Back_modelsel" + binkey + sigfunc + bkgfunc, binInfo + ";Model_cut;Expected Significance #times BDT Efficiency", 1,-10,15, 1,0,1.1*np.max(significanceMulBDTEff) )
                    gr_ModelSelection = ROOT.TGraph(len(MODEL_Eff_LIST), modelOutput_x, significanceMulBDTEff)
                    h_Back_modelsel.Draw()
                    gr_ModelSelection.SetLineColor(ROOT.kBlue)
                    gr_ModelSelection.Draw("L same")
                    saveptdir.WriteObject(c_ModelSelection, "ModelSelection" + binkey + sigfunc + bkgfunc)
                    
                    maxindex=np.argmax(significanceMulBDTEff)
                    #variation of corrected signal counts
                    c_SigToBDTefficiency = myH.TCanvas()
                    c_SigToBDTefficiency.cd()
                    SigToBDTefficiency_y = np.array(signalCount[icent][ipt][isig][ibkg])/BDTefficiency_y
                    SigToBDTefficiency_errory = np.array(signalError[icent][ipt][isig][ibkg])/BDTefficiency_y
                    gr_SigToBDTefficiency = ROOT.TGraphErrors(len(MODEL_Eff_LIST), np.array(MODEL_Eff_LIST), SigToBDTefficiency_y, np.zeros(len(MODEL_Eff_LIST)), SigToBDTefficiency_errory )
                    gr_SigToBDTefficiency1 = ROOT.TGraphErrors(len(MODEL_Eff_LIST), np.array(MODEL_Eff_LIST), SigToBDTefficiency_y)
                    tf1_sigtoBDTeff = ROOT.TF1("tf1_sigtoBDTeff","[0]",max(MODEL_Eff_LIST[0],0.01*maxindex),min(0.01*maxindex+0.2,MODEL_Eff_LIST[-1]))
                    tf1_sigtoBDTeff.SetParName(0,"mean")
                    tf1_sigtoBDTeff.SetLineColor(ROOT.kRed)
                    tf1_sigtoBDTeff.SetLineStyle(2)
                    tf1_sigtoBDTeff.SetLineWidth(2)
                    gr_SigToBDTefficiency.SetTitle(binInfo + ';Eff_{BDT};N_{Sig} / Eff_{BDT}')
                    gr_SigToBDTefficiency.SetFillColor(ROOT.kCyan-10)
                    gr_SigToBDTefficiency.Draw("3A")
                    gr_SigToBDTefficiency1.SetLineColor(4)
                    gr_SigToBDTefficiency1.SetMarkerStyle(8)
                    gr_SigToBDTefficiency1.SetMarkerSize(0.4)
                    #gr_SigToBDTefficiency1.Fit("tf1_sigtoBDTeff","R")
                    gr_SigToBDTefficiency1.Draw("same LP")
                    saveptdir.WriteObject(c_SigToBDTefficiency, "SigToBDTefficiency" + binkey + sigfunc + bkgfunc)
                    
                    #variation of BDT efficiency selection
                    for var in range(-10, 11):
                        if (maxindex + var >= len(MODEL_Eff_LIST)) or (maxindex + var < 0):
                            continue
                        CorrectedSignal.append( SigToBDTefficiency_y[maxindex+var]/(Efficiency[icent][ipt] * AbsorbFactor[icent][ipt]) )
            #Cal Systematic uncertainties
            binInfo = "pT" + str(ptbin[0]) + "-" + str(ptbin[1]) + " GeV/c"     
            c_CorrectedSignal = myH.TCanvas()
            c_CorrectedSignal.cd()
            CorrectedSignal_y = np.array(CorrectedSignal)
            h_CorrectedSignal = ROOT.TH1F("hCorrectedSignal" + binkey, binInfo + ";Corrected Hypertriton signal;Counts", 20, 0.9*np.min(CorrectedSignal_y), 1.1*np.max(CorrectedSignal_y) )
            for sigVar in CorrectedSignal:
                h_CorrectedSignal.Fill(sigVar)
            h_CorrectedSignal.Draw()
            saveptdir.WriteObject(c_CorrectedSignal, "CorrectedSignal" + binkey)

            #Use the last case of sig and bkg func as central value to calculate the uncertainty (now we only use the rms)
            isig = len(sigpdf) - 1
            ibkg = len(bkgpdf) - 1
            #SystUnc[icent].append( math.sqrt( math.pow(signalCount[icent][ipt][isig][ibkg][maxindex]/(Efficiency[icent][ipt] * BDTefficiency_y[maxindex] * AbsorbFactor[icent][ipt])  - h_CorrectedSignal.GetMean(), 2) + math.pow(h_CorrectedSignal.GetStdDev(), 2) ) )
            SystUnc[icent].append(h_CorrectedSignal.GetStdDev() )
            print("SysUnc Check(dev and rms):", round(signalCount[icent][ipt][isig][ibkg][maxindex]/(Efficiency[icent][ipt] * BDTefficiency_y[maxindex] * AbsorbFactor[icent][ipt]) - h_CorrectedSignal.GetMean(),2), round(h_CorrectedSignal.GetStdDev(),2))
            
            CorrectedYield_y = np.array(CorrectedSignal)/(fRead.getEventNumber(dataType) * 0.4 * 2 * (ptbin[1] - ptbin[0]))
            if ifTellMatter:
                CorrectedYield_y = CorrectedYield_y / 2
            h_CorrectedYield = ROOT.TH1F("hCorrectedYield" + binkey, binInfo + ";Corrected Hypertriton yield;Counts", 20, 0.8*np.min(CorrectedYield_y), 1.2*np.max(CorrectedYield_y) )
            for sigVar in CorrectedYield_y:
                h_CorrectedYield.Fill(sigVar)
            saveptdir.WriteObject(h_CorrectedYield, "hCorrectedYield" + binkey)
            
            # New tree to store the corrected yield in the range of 3 sigma
            mean = h_CorrectedYield.GetMean()
            rms  = h_CorrectedYield.GetRMS()
            xTTree = np.zeros(1, dtype=np.float64)
            h_CorrectedYieldTree = ROOT.TH1F("hCorrectedYieldTree" + binkey, binInfo + ";Corrected Hypertriton yield;Counts", 20, 0.8*np.min(CorrectedYield_y), 1.2*np.max(CorrectedYield_y) )
            CorrectedYieldTree = ROOT.TTree("CorrectedYield"+binkey, "CorrectedYield Tree")
            CorrectedYieldTree.Branch("CorrectedYield",xTTree,'CorrectedYield/D')
            for sigVar in CorrectedYield_y:
                if abs(sigVar - mean) <= 3 * rms:
                    xTTree[0] = sigVar
                    h_CorrectedYieldTree.Fill(sigVar)
                    CorrectedYieldTree.Fill()
            saveptdir.WriteObject(h_CorrectedYieldTree, "hCorrectedYieldTree" + binkey)
            saveptdir.WriteObject(CorrectedYieldTree, "CorrectedYield" + binkey)
    print("Systematical Uncertainty finish")

# Calculate Yield
HypYields = myH.createEmptyList( [len(CENT_BIN_LIST)] )
HypYieldsError = myH.createEmptyList( [len(CENT_BIN_LIST)] )
SystYieldsError = myH.createEmptyList( [len(CENT_BIN_LIST)] )
ModelIndex = myH.createEmptyList( [len(CENT_BIN_LIST)] )
#Use the last case of sig and bkg func as central value
isig = len(sigpdf) - 1
ibkg = len(bkgpdf) - 1
for icent, centbin in enumerate(CENT_BIN_LIST):
    print("Cent " + str(centbin[0]) + "-" + str(centbin[1]) + "%")
    for ipt, ptbin in enumerate(PT_BIN_LIST[icent]):
        # binkey = "Cent" + str(centbin[0]) + "_" + str(centbin[1]) + "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
        binkey = "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
        # binkey = "ModelTest"
        significance = np.array(ExpSignificance[icent][ipt][isig][ibkg])
        BDTefficiency = score_eff_arrays_dict[binkey][1][0:len(MODEL_Eff_LIST)]
        significanceMulBDTEff = significance * BDTefficiency
        maxindex=np.argmax(significanceMulBDTEff)
        ModelIndex[icent].append(maxindex)
        print(round(MODEL_Eff_LIST[maxindex],2), score_eff_arrays_dict[binkey][0][maxindex])
        
        if ifTellMatter:
            HypYields[icent].append(signalCount[icent][ipt][isig][ibkg][maxindex] / (0.4 * fRead.getEventNumber(dataType) * Efficiency[icent][ipt] * BDTefficiency[maxindex] * AbsorbFactor[icent][ipt] * (ptbin[1] - ptbin[0]) ))
            HypYieldsError[icent].append(signalError[icent][ipt][isig][ibkg][maxindex] / (0.4 * fRead.getEventNumber(dataType) * Efficiency[icent][ipt] * BDTefficiency[maxindex] * AbsorbFactor[icent][ipt] * (ptbin[1] - ptbin[0]) ) )
            SystYieldsError[icent].append( math.sqrt( math.pow( SystUnc[icent][ipt] / (0.4 * fRead.getEventNumber(dataType) * (ptbin[1] - ptbin[0]) ), 2) + math.pow( HypYields[icent][ipt] * (1-AbsorbFactor[icent][ipt]) * 0.5, 2) ) )
        else:
            HypYields[icent].append(signalCount[icent][ipt][isig][ibkg][maxindex] / (0.4 * 2 * fRead.getEventNumber(dataType) * Efficiency[icent][ipt] * BDTefficiency[maxindex] * AbsorbFactor[icent][ipt] * (ptbin[1] - ptbin[0]) ))
            HypYieldsError[icent].append(signalError[icent][ipt][isig][ibkg][maxindex] / (0.4 * 2 * fRead.getEventNumber(dataType) * Efficiency[icent][ipt] * BDTefficiency[maxindex] * AbsorbFactor[icent][ipt] * (ptbin[1] - ptbin[0]) ) )
            SystYieldsError[icent].append( math.sqrt( math.pow( SystUnc[icent][ipt] / (0.4 * 2 * fRead.getEventNumber(dataType) * (ptbin[1] - ptbin[0]) ), 2) + math.pow( HypYields[icent][ipt] * (1-AbsorbFactor[icent][ipt]) * 0.5, 2) ) )

# Readin hypertriton yield from 2-body decay analysis
f = ROOT.TFile("../CC_file/pt_analysis_antimat_2024_newcut.root", "READ")
h_correct_counts = f.Get("std/h_corrected_counts")
hStat = f.Get("std/hStat")
hSystRMS = f.Get("std/hSystRMS")
rawyield_2body = []
statunc_2body = []
systunc_2body = []
statE = np.zeros(1)
systE = np.zeros(1)
index_bins = [[2, 2], [3, 5], [6, 7]]
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

#Save final results
with ROOT.TFile(savePrefix + "Results.root", "recreate") as outfile:
    for icent, centbin in enumerate(CENT_BIN_LIST):
        Pt_x = np.array([])
        Pt_errorx = np.array([])
        for ipt, ptbin in enumerate(PT_BIN_LIST[icent]):
            Pt_x = np.append( Pt_x, (ptbin[0] + ptbin[1])/2 )
            Pt_errorx = np.append( Pt_errorx, (ptbin[1] - ptbin[0])/2 )
            maxindex = ModelIndex[icent][ipt]
            print(maxindex)
            # binkey = "ModelCent" + str(centbin[0]) + "_" + str(centbin[1]) + "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
            binkey =  "pT" + str(ptbin[0]) + "_" + str(ptbin[1])
            model_threshold = score_eff_arrays_dict[binkey][0][maxindex]
            name =  "pT" + str(ptbin[0]) + "_" + str(ptbin[1]) + "_" + dataType
            (signalN, bkgN, paras) = data.invMFit(icent, ipt, sigfunc, bkgfunc, model_threshold, name=name, para=MCFitpara[icent][ipt][isig], ifDebug=True, outfile = outfile)

        # Ratio of B.R.
        y_3body =np.array(HypYields[icent]) * 0.4
        dy = np.array(HypYieldsError[icent]) * 0.4
        dysyst = np.array(SystYieldsError[icent]) * 0.4
        gr_R_y = rawyield_2body / (rawyield_2body + y_3body)
        gr_R_errory = np.sqrt(y_3body**2 * statunc_2body**2 + rawyield_2body**2 * dy**2) / (rawyield_2body + y_3body)**2 
        gr_R_errorysyst = np.sqrt(y_3body**2 * systunc_2body**2 + rawyield_2body**2 * dysyst**2) / (rawyield_2body + y_3body)**2
        gr_R_errorysyst = np.zeros(Pt_x.size)
        c_R = myH.TCanvas('c_R','c_R')
        c_R.cd()
        gr_R = ROOT.TGraphMultiErrors(Pt_x.size, Pt_x, gr_R_y, Pt_errorx, Pt_errorx, gr_R_errory, gr_R_errory)
        gr_R.AddYError(Pt_x.size, gr_R_errorysyst, gr_R_errorysyst)
        gr_R.SetTitle("")
        gr_R.GetXaxis().SetTitle( '#it{p}_{T} (GeV/c)' )
        gr_R.GetYaxis().SetTitle( 'R' )
        gr_R.GetAttLine(0).SetLineColor(1)
        gr_R.GetAttLine(1).SetLineColor(4)
        gr_R.GetAttFill(1).SetFillStyle(0)
        gr_R.SetMarkerColor(4)
        gr_R.SetMarkerStyle(8)
        gr_R.SetMarkerSize(1.5)
        tf1_R = ROOT.TF1("tf1_R","[0]", 2, 5)
        gr_R.Fit("tf1_R")
        gr_R.Draw("APS ; Z ; 5 s=0.5")
        outfile.WriteObject(c_R, 'R')

        # Yield by assuming B.R. = 0.4
        gr_HypYields_y = np.array(HypYields[icent])
        gr_HypYields_errory = np.array(HypYieldsError[icent])
        gr_HypYields_errorysyst = np.array(SystYieldsError[icent])
        c_HypYields = myH.TCanvas('HypYields','HypYields')
        c_HypYields.cd()
        h_Back_Yields = ROOT.TH2F("h_Back_Yields", ";#it{p}_{T} (GeV/c);Yields", 9, 1, 10, 10,0, 3*math.pow(10, -5))
        gr_HypYields = ROOT.TGraphMultiErrors(Pt_x.size, Pt_x, gr_HypYields_y, Pt_errorx, Pt_errorx, gr_HypYields_errory, gr_HypYields_errory)
        gr_HypYields.AddYError(Pt_x.size, gr_HypYields_errorysyst, gr_HypYields_errorysyst)
        gr_HypYields.SetTitle("")
        gr_HypYields.GetXaxis().SetTitle( '#it{p}_{T} (GeV/c)' )
        gr_HypYields.GetYaxis().SetTitle( 'Yields' )
        gr_HypYields.GetAttLine(0).SetLineColor(1)
        gr_HypYields.GetAttLine(1).SetLineColor(4)
        gr_HypYields.GetAttFill(1).SetFillStyle(0)
        gr_HypYields.SetMarkerColor(4)
        gr_HypYields.SetMarkerStyle(8)
        gr_HypYields.SetMarkerSize(1.5)
        h_Back_Yields.Draw("")
        gr_HypYields.Draw("APS ; Z ; 5 s=0.5")
        pt_spectrum.Draw("same")
        outfile.WriteObject(c_HypYields, 'HypYields')
