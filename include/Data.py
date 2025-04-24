import utils
import simultaneous_fit as emFit
import numpy as np
from array import array
import ROOT
from copy import deepcopy

############### Data Interval ###############
class dataInterval:
    'data frame in specific centrality and pt bin'

    def __init__(self, df, preCut=None, cent_bin=[-999, 999], pt_bin=[-999, 999], MASS_BIN=[2.96, 3.04], dataType="", model=None):
        self.cent_bin = cent_bin
        self.pt_bin = pt_bin
        self.MASS_BIN = MASS_BIN
        self.dataType = dataType
        self.model = model
        self.signal_num = None
        self.signal_error = None
        self.exp_bkg_num = None

        cut = (
            f'({cent_bin[0]} <= fCentrality < {cent_bin[1]}) and '
            f'({pt_bin[0]} <= fPt < {pt_bin[1]}) and '
            f'({MASS_BIN[0]} <= fM < {MASS_BIN[1]})'
        )
        if preCut is not None:
            cut = cut + " and " + preCut

        self.dataPD = deepcopy(df.query(cut))

    def getDF(self):
        return self.dataPD
    
    def setModel(self, model):
        self.model = model
    
    def addModelPrediction(self, output_margin=True):
        if self.model is None:
            raise ValueError("BDT Model not set.")
        score = self.model.predict(self.dataPD, output_margin)
        self.dataPD["model_output"] = score
    
    def invMFit(self, sigpdf="Gauss", bkgpdf="exp", model_threshold=None, name=None, para=None, isMC=False, ifDrawStats=True, ifDebug=False,outfile=None):
        ''' Fit the invariant mass distribution,'''

        title = " pT {}-{} GeV/c {}".format(self.pt_bin[0], self.pt_bin[1], self.dataType)

        pddata = self.dataPD
        if model_threshold is not None:
            pddata = pddata.query(f'model_output > {model_threshold}')

        #calculate efficiency for each pt bin without ML model selection
        xframe, signal_num, bkg_num, paras = utils.fitInvMass(
            pddata, title, self.MASS_BIN, nbins=0, sigpdf=sigpdf, bkgpdf=bkgpdf,
            para=para, isMC=isMC, ifDrawStats=ifDrawStats, ifDebug=ifDebug
        )

        if outfile is not None:
            C = utils.TCanvas("C", "C", 800, 600)
            xframe.SetName("pT " + str(self.pt_bin[0]) + "_" + str(self.pt_bin[1]) + self.dataType)
            xframe.Draw()
            xframe.SetDirectory(ROOT.nullptr)
            if not name:
                name = title
            outfile.WriteObject(C, name)
            del C

        del xframe
        return(signal_num, bkg_num, paras)
    
    def simultaneousFit(self, bkg_deuteron, bkg_uncorrelated, sigpdf, corr_bkgpdf, uncorr_bkgpdf, mcpara=None, fit_massbin=[2.96, 3.02], model_eff=1, model_threshold=None, outfile=None, ifDebug=False):
        binInfo = f" pT {self.pt_bin[0]}-{self.pt_bin[1]} GeV/c BDTEff={round(model_eff, 2)}"

        if model_eff != 1:
            data_se = self.dataPD.query(f'model_output>{model_threshold}')
            bkg_me_deuteron = bkg_deuteron.dataPD.query(f'model_output>{model_threshold}')
            bkg_me_uncorrelated = bkg_uncorrelated.dataPD.query(f'model_output>{model_threshold}')
        else:
            data_se = self.dataPD
            bkg_me_deuteron = bkg_deuteron.dataPD
            bkg_me_uncorrelated = bkg_uncorrelated.dataPD

        if len(data_se) == 0:
            return 0, 0.1, 999

        print("Fitting ", binInfo, " with ", sigpdf, " and ", corr_bkgpdf)

        hRawYield, canvas_bkg, canvas_signal, signal_num, signal_error, exp_bkg_num, bkg_peak_value = emFit.simultaneousFit(
            data_se, bkg_me_deuteron, bkg_me_uncorrelated,
            mcpara, nBins=35, ptlims=self.pt_bin, lowMassLim=fit_massbin[0], highMassLim=fit_massbin[1],
            title=binInfo, signalPdf=sigpdf, corr_bkgPdf=corr_bkgpdf, uncorr_bkgPdf=uncorr_bkgpdf,
            df_column="fM", corr_bkg_peak=None
        )

        if outfile is not None:
            outfile.cd()
            canvas_signal.Write()
            canvas_bkg.Write()
        del hRawYield, canvas_bkg, canvas_signal

        return(signal_num, signal_error, exp_bkg_num)

############### MC Data Interval ############### 
class mcDataInterval(dataInterval):
    'MC data frame in specific centrality and pt bin'

    def __init__(self, df, preCut = None, cent_bin = [-999, 999], pt_bin = [-999, 999], MASS_BIN = [2.96, 3.04], dataType = "", model=None):
        super().__init__(df, preCut, cent_bin, pt_bin, MASS_BIN, dataType, model)
        self.MCGenSignalCount = None
        self.MCRecSignalCount = None
        self.Efficiency = None
        self.ExpCorrectedSignal = None
    
    def calculateEfficiency(self, MCGenPD, pt_spectrum, eventNumber, BR, f_absorption=1.0, is_single_matter_type=False):
        '''Calculate the efficiency, expected yield and corrected signal number'''

        binCut = f'(fGenPt >= {self.pt_bin[0]}) and (fGenPt < {self.pt_bin[1]})'
        self.MCGenSignalCount = len(MCGenPD.query(f"fIsSignal == 1 and abs(fGenRapidity) < 0.5 and {binCut}"))
        self.MCRecSignalCount = len(self.dataPD)

        if self.MCGenSignalCount > 0:
            self.Efficiency = self.MCRecSignalCount / self.MCGenSignalCount
        else:
            self.Efficiency = 0
        print("Efficiency in ", f'( {self.cent_bin[0]} - {self.cent_bin[1]} ) and ( {self.pt_bin[0]} <= pT < {self.pt_bin[1]} ), :', self.Efficiency)
        print("Expected Yield", pt_spectrum.Integral(self.pt_bin[0], self.pt_bin[1]) / (self.pt_bin[1] - self.pt_bin[0]) )

        corrfactor = self.Efficiency * f_absorption * BR * eventNumber
        expsig = 2 * pt_spectrum.Integral(self.pt_bin[0], self.pt_bin[1]) * corrfactor
        if is_single_matter_type:
            expsig = expsig / 2
        self.ExpCorrectedSignal = expsig
        print("Expected SignalYield:", expsig)

############### Data Group ###############
class dataGroup:
    'Object for local analysis'

    def __init__(self, df, preCut=None, CENT_BIN_LIST=[[-999, 999]], PT_BIN_LIST=[[[-999, 999]]], MASS_BIN=[2.96, 3.04], dataType="", modelList=None):
        self._build_data(df, preCut, CENT_BIN_LIST, PT_BIN_LIST, MASS_BIN, dataType, modelList=modelList)
    
    def _build_data(self, df, preCut=None, CENT_BIN_LIST=[[-999, 999]], PT_BIN_LIST=[[[-999, 999]]], MASS_BIN=[2.96, 3.04], dataType="", modelList=None, isMC=False):
        self.data = []
        self.CENT_BIN_LIST = CENT_BIN_LIST
        self.PT_BIN_LIST = PT_BIN_LIST
        self.MASS_BIN = MASS_BIN
        self.hist_ptBins = []
        self.dataType = dataType
        self.paras = None
        self.signalCount = None
        self.signalError = None
        self.expBkgCount = None

        for icent, cent_bin in enumerate(CENT_BIN_LIST):
            pt_bins = np.array([pt[0] for pt in PT_BIN_LIST[icent]] + [PT_BIN_LIST[icent][-1][1]])
            pt_bins = array('d', pt_bins.astype(np.float64))
            self.hist_ptBins.append(pt_bins)

            self.data.append([])
            for ipt, pt_bin in enumerate(PT_BIN_LIST[icent]):
                if isMC:
                    data = mcDataInterval(df, preCut, cent_bin, pt_bin, MASS_BIN, dataType)
                else:
                    data = dataInterval(df, preCut, cent_bin, pt_bin, MASS_BIN, dataType)
                if modelList is not None:
                    data.setModel(modelList[icent][ipt])
                self.data[icent].append(deepcopy(data))

    def getDataInterval(self, icent, ipt):
        return self.data[icent][ipt]

    def getDF(self, icent, ipt):
        return self.data[icent][ipt].getDF()
    
    def setModel(self, modelList):
        for icent, cent_bin in enumerate(self.CENT_BIN_LIST):
            for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
                self.data[icent][ipt].setModel(modelList[icent][ipt])

    def addModelPrediction(self, output_margin=True):
        for icent, cent_bin in enumerate(self.CENT_BIN_LIST):
            for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
                self.data[icent][ipt].addModelPrediction(output_margin)

    def invMFit(self, icent, ipt, sigpdf="Gauss", bkgpdf="exp", model_threshold=None,name=None, para=None, isMC=False, ifDrawStats=True, ifDebug=False,outfile=None):
        ''' Fit the invariant mass distribution,'''
        return self.data[icent][ipt].invMFit(
            sigpdf=sigpdf,
            bkgpdf=bkgpdf,
            model_threshold=model_threshold,
            name=name,
            para=para,
            isMC=isMC,
            ifDrawStats=ifDrawStats,
            ifDebug=ifDebug,
            outfile=outfile
        )
    
    def doSimultaneousFits(self, bkg_deuteron, bkg_uncorrelated, sigpdf_list, bkgpdf_list, MODEL_Eff_LIST, score_eff_arrays_dict,
                           mcpara_list=None, fit_massbin=[2.96, 3.02], outfile=None, ifDebug=False):
        signalCount = utils.createEmptyList( [len(self.CENT_BIN_LIST), len(self.PT_BIN_LIST[0]), len(sigpdf_list), len(bkgpdf_list)] )
        expBkgCount = utils.createEmptyList( [len(self.CENT_BIN_LIST), len(self.PT_BIN_LIST[0]), len(sigpdf_list), len(bkgpdf_list)] )
        signalError = utils.createEmptyList( [len(self.CENT_BIN_LIST), len(self.PT_BIN_LIST[0]), len(sigpdf_list), len(bkgpdf_list)] )
        for icent, cent_bin in enumerate(self.CENT_BIN_LIST):
            outdir_cent = outfile.mkdir("Cent"+str(cent_bin[0]) + "_" + str(cent_bin[1]))
            for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
                outdir_pt = outdir_cent.mkdir("pT"+str(pt_bin[0]) + "_" + str(pt_bin[1]))
                binkey =  "pT" + str(pt_bin[0]) + "_" + str(pt_bin[1])
                data_se = self.getDataInterval(icent, ipt)
                bkg_me_deuteron = bkg_deuteron.getDataInterval(icent, ipt)
                bkg_me_uncorrelated = bkg_uncorrelated.getDataInterval(icent, ipt)
                for isig, sigfunc in enumerate(sigpdf_list):
                    outdir_sig = outdir_pt.mkdir(sigfunc)
                    for ibkg, bkgfunc in enumerate(bkgpdf_list):
                        outdir_bkg = outdir_sig.mkdir(bkgfunc)
                        for imodel, model_eff in enumerate(MODEL_Eff_LIST):
                            if model_eff != 1:
                                model_threshold = score_eff_arrays_dict[binkey][0][imodel]
                            else:
                                model_threshold = None
                            (signal_num, signal_error, exp_bkg_num) = data_se.simultaneousFit(bkg_me_deuteron, bkg_me_uncorrelated, sigfunc, corr_bkgpdf=bkgfunc, uncorr_bkgpdf="pol1", 
                                                                            mcpara=mcpara_list[icent][ipt][0], fit_massbin=fit_massbin, model_eff=model_eff, model_threshold=model_threshold,
                                                                            outfile=outdir_bkg)
                            signalCount[icent][ipt][isig][ibkg].append(signal_num)
                            signalError[icent][ipt][isig][ibkg].append(signal_error)
                            expBkgCount[icent][ipt][isig][ibkg].append(exp_bkg_num)
        self.signalCount = signalCount
        self.signalError = signalError
        self.expBkgCount = expBkgCount
        return signalCount, signalError, expBkgCount
    
############### MC Data Group ###############
class mcDataGroup(dataGroup):
    'MC data frame in specific centrality and pt bin'

    def __init__(self, df, preCut = None, CENT_BIN_LIST = [[-999, 999]], PT_BIN_LIST = [[[-999, 999]]], MASS_BIN = [2.96, 3.04], dataType = "", modelList=None):
        self._build_data(df, preCut, CENT_BIN_LIST, PT_BIN_LIST, MASS_BIN, dataType, modelList=modelList, isMC=True)

    def calculateEfficiency(self, MCGenPD, pt_spectrum, eventNumber, BR, absorbfactor_list=None, is_single_matter_type=False):
        for icent, cent_data in enumerate(self.data):
            for ipt, data_interval in enumerate(cent_data):
                f_absorption = 1.0
                if absorbfactor_list is not None:
                    f_absorption = absorbfactor_list[icent][ipt]
                data_interval.calculateEfficiency(MCGenPD, pt_spectrum, eventNumber, BR, f_absorption=f_absorption, is_single_matter_type=is_single_matter_type)

    def saveEfficiencyPlots(self, outfile):
        for icent, cent_data in enumerate(self.data):
            c = utils.TCanvas(f"c_eff_Cent{self.CENT_BIN_LIST[icent][0]}_{self.CENT_BIN_LIST[icent][1]}", "Efficiency vs pT")

            hMCGenSignalCount = ROOT.TH1F(f"hMCGenSignalCount_Cent{icent}", "hMCGenSignalCount", len(self.hist_ptBins[icent])-1, self.hist_ptBins[icent])
            hEff = ROOT.TH1F(f"hEff_Cent{icent}", ";p_{T} (GeV/c);Precut Efficiency * Acceptance", len(self.hist_ptBins[icent])-1, self.hist_ptBins[icent])

            for ipt, data_interval in enumerate(cent_data):
                hMCGenSignalCount.SetBinContent(ipt+1, data_interval.MCGenSignalCount)
                hMCGenSignalCount.SetBinError(ipt+1, 0)
                hEff.SetBinContent(ipt+1, data_interval.MCRecSignalCount)

            hEff.Divide(hMCGenSignalCount)
            hEff.SetMarkerStyle(8)
            hEff.SetMarkerSize(0.5)
            hEff.GetYaxis().SetRangeUser(0, 0.08)
            hEff.Draw("ep")

            outfile.cd()
            c.Write()