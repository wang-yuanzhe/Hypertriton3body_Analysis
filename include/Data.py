import myHeader as myH
import numpy as np
import ROOT
from copy import deepcopy

class analysisData:
    'Object for local analysis'

    def __init__(self, df, preCut = "", CENT_BIN_LIST = [[-999, 999]], PT_BIN_LIST = [[[-999, 999]]], MASS_BIN = [2.96, 3.04], dataType = ""):
        ''' Initialize the dataset list and corresponding bins'''

        self.dataPD = []
        self.CENT_BIN_LIST = CENT_BIN_LIST
        self.PT_BIN_LIST = PT_BIN_LIST
        self.MASS_BIN = MASS_BIN
        self.hist_ptBins = []
        self.dataType = dataType
        self.paras = None
        for icent, centbin in enumerate(CENT_BIN_LIST):
            self.hist_ptBins.append(np.array([]))
            self.dataPD.append([])
            for ptbin in PT_BIN_LIST[icent]:
                self.hist_ptBins[icent] = np.append(self.hist_ptBins[icent], ptbin[0])
                cut = f'( {centbin[0]} <= fCentrality < {centbin[1]} ) and ( {ptbin[0]} <= fPt < {ptbin[1]} ) and ( {MASS_BIN[0]} <= fM < {MASS_BIN[1]} )'
                if preCut != "":
                    cut = cut + " and " + preCut
                self.dataPD[icent].append(deepcopy(df.query(cut)))
            self.hist_ptBins[icent] = np.append(self.hist_ptBins[icent], PT_BIN_LIST[icent][-1][1])

    def getDF(self, icent, ipt):
        return self.dataPD[icent][ipt]

    def addModelPrediction(self, model, output_margin=True):
        '''Add model predictions to the dataframes'''

        for icent, centbin in enumerate(self.CENT_BIN_LIST):
            for ipt, ptbin in enumerate(self.PT_BIN_LIST[icent]):
                score = model[icent][ipt].predict(self.dataPD[icent][ipt], output_margin=output_margin)
                self.dataPD[icent][ipt]["model_output"] = score

    def invMFit(self, icent, ipt, sigpdf="Gauss", bkgpdf="exp", model_threshold=None,name=None, para=None, isMC=False, ifDrawStats=True, ifDebug=False,outfile=None):
        ''' Fit the invariant mass distribution,'''

        ptbin = self.PT_BIN_LIST[icent][ipt]
        title = " pT " + str(ptbin[0]) + "-" + str(ptbin[1]) + " GeV/c " + self.dataType + " " + sigpdf
        if model_threshold:
            pddata = self.dataPD[icent][ipt].query(f'model_output>{model_threshold}')
        else:
            pddata = self.dataPD[icent][ipt]
        #calculate efficiency for each pt bin without ML model selection
        (xframe, signalN, bkgN, paras) = myH.fitInvMass(pddata, title, self.MASS_BIN, nbins = 0, sigpdf=sigpdf, bkgpdf=bkgpdf, para=para, isMC=isMC, ifDrawStats=ifDrawStats, ifDebug=ifDebug, Matter = 0)
        if outfile != None:
            C = myH.TCanvas("C", "C", 800, 600)
            xframe.SetName("pT " + str(ptbin[0]) + "_" + str(ptbin[1]) + self.dataType)
            xframe.Draw()
            xframe.SetDirectory(ROOT.nullptr)
            if not name:
                name = title
            outfile.WriteObject(C, name)
        del xframe, C
        return(signalN, bkgN, paras)