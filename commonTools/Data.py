import utils
import simultaneous_fit as emFit
import numpy as np
from array import array
import ROOT
import math
from copy import deepcopy

############### Data Interval ###############
class DataInterval:
    'data frame in specific centrality and pt bin'

    def __init__(self, df, pre_cut = None, cent_bin = [-999, 999], pt_bin = [-999, 999], MASS_BIN = [2.96, 3.04], dataType = "", is_single_matter_type = False, model = None):
        self.cent_bin = cent_bin
        self.pt_bin = pt_bin
        self.MASS_BIN = MASS_BIN
        self.dataType = dataType
        self.is_single_matter_type = is_single_matter_type
        self.model = model
        self.signal_num = None
        self.signal_error = None
        self.exp_bkg_num = None

        cut = (
            f'({cent_bin[0]} <= fCentrality < {cent_bin[1]}) and '
            f'({pt_bin[0]} <= fPt < {pt_bin[1]}) and '
            f'({MASS_BIN[0]} <= fM < {MASS_BIN[1]})'
        )
        if pre_cut is not None:
            cut = cut + f" and {pre_cut}"

        self.dataPD = deepcopy(df.query(cut))

    def getCorrectedSignalCount(self, icent, ipt, f_absorption, imodel, f_corr):
        if self.signal_num is None:
            raise ValueError("Signal number not set.")
        
        # f_precut_corr = Branching Ratio * Event Number
        corrfactor = f_corr * self.precut_efficiency[icent][ipt] * self.bdt_efficiency[imodel] * f_absorption * (self.pt_bin[1] - self.pt_bin[0])
        if not self.is_single_matter_type:
            corrfactor = corrfactor * 2

        return self.signal_num / corrfactor

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
            xframe.SetName(f"pT{self.pt_bin[0]}_{self.pt_bin[1]}_{self.dataType}")
            xframe.Draw()
            xframe.SetDirectory(ROOT.nullptr)
            if not name:
                name = title
            outfile.WriteObject(C, name)
            del C

        del xframe
        return(signal_num, bkg_num, paras)
    
    def simultaneousFit(self, bkg_deuteron, bkg_uncorrelated, sigpdf, corr_bkgpdf, uncorr_bkgpdf, mcpara=None, fit_massbin=[2.96, 3.02], model_eff=1, model_threshold=None, outfile=None, corr_bkg_peak=None, fix_bkg_peak = True, ifDebug=False):
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
            df_column="fM", corr_bkg_peak=corr_bkg_peak, fix_bkg_peak=fix_bkg_peak
        )

        if outfile is not None:
            outfile.cd()
            canvas_signal.Write()
            canvas_bkg.Write()
        del hRawYield, canvas_bkg, canvas_signal

        return(signal_num, signal_error, exp_bkg_num, bkg_peak_value)

############### MC Data Interval ############### 
class mcDataInterval(DataInterval):
    'MC data frame in specific centrality and pt bin'

    def __init__(self, df, pre_cut = None, cent_bin = [-999, 999], pt_bin = [-999, 999], MASS_BIN = [2.96, 3.04], dataType = "", is_single_matter_type=False, model=None):
        super().__init__(df, pre_cut, cent_bin, pt_bin, MASS_BIN, dataType, is_single_matter_type, model)
        self.MCGenSignalCount = None
        self.MCRecSignalCount = None
        self.Efficiency = None
        self.ExpCorrectedSignal = None
    
    def calculateEfficiency(self, MCGenPD, pt_spectrum, eventNumber_in_realdata, BR, f_absorption=1.0, is_single_matter_type=False):
        '''Calculate the efficiency, expected yield and corrected signal number'''

        binCut = f'(fGenPt >= {self.pt_bin[0]}) and (fGenPt < {self.pt_bin[1]})'
        self.MCGenSignalCount = len(MCGenPD.query(f"fIsSignal == 1 and abs(fGenRapidity) < 0.5 and {binCut}"))
        self.MCRecSignalCount = len(self.dataPD)

        self.Efficiency = self.MCRecSignalCount / self.MCGenSignalCount if self.MCGenSignalCount > 0 else 0
        print("Efficiency in ", f'( {self.cent_bin[0]} - {self.cent_bin[1]} ) and ( {self.pt_bin[0]} <= pT < {self.pt_bin[1]} ), :', self.Efficiency)
        print("Expected Yield", pt_spectrum.Integral(self.pt_bin[0], self.pt_bin[1]) / (self.pt_bin[1] - self.pt_bin[0]) )

        corrfactor = self.Efficiency * f_absorption * BR * eventNumber_in_realdata
        if not is_single_matter_type:
            corrfactor = corrfactor * 2
        expsig = pt_spectrum.Integral(self.pt_bin[0], self.pt_bin[1]) * corrfactor
        self.ExpCorrectedSignal = expsig
        print("Expected SignalYield:", expsig)
        return self.Efficiency, self.ExpCorrectedSignal

############### Data Group ###############
class DataGroup:
    'Object for local analysis'

    def __init__(self, df, pre_cut = None, CENT_BIN_LIST = [[-999, 999]], PT_BIN_LIST = [[[-999, 999]]], MASS_BIN = [2.96, 3.04], SIGPDF_LIST = ["dscb"], BKGPDF_LIST = ["expo"], dataType = "", is_single_matter_type = False, model_list = None):
        self._build_data(df, pre_cut, CENT_BIN_LIST, PT_BIN_LIST, MASS_BIN, SIGPDF_LIST, BKGPDF_LIST, dataType, is_single_matter_type=is_single_matter_type, model_list=model_list)
    
    def _build_data(self, df, pre_cut = None, CENT_BIN_LIST = [[-999, 999]], PT_BIN_LIST = [[[-999, 999]]], MASS_BIN = [2.96, 3.04], SIGPDF_LIST = ["dscb"], BKGPDF_LIST = ["expo"], dataType = "", is_single_matter_type = False, model_list = None, isMC = False):
        self.data = []
        self.CENT_BIN_LIST = CENT_BIN_LIST
        self.PT_BIN_LIST = PT_BIN_LIST
        self.SIGPDF_LIST = SIGPDF_LIST
        self.BKGPDF_LIST = BKGPDF_LIST
        self.MASS_BIN = MASS_BIN
        self.hist_ptBins = []
        self.dataType = dataType
        self.is_single_matter_type = is_single_matter_type
        self.paras = None
        self.signalCount = None
        self.signalError = None
        self.expBkgCount = None
        self.precut_efficiency = None
        self.bdt_efficiency = None
        self.bestBDTIndex = None

        for icent, cent_bin in enumerate(CENT_BIN_LIST):
            pt_bins = np.array([pt[0] for pt in PT_BIN_LIST[icent]] + [PT_BIN_LIST[icent][-1][1]])
            pt_bins = array('d', pt_bins.astype(np.float64))
            self.hist_ptBins.append(pt_bins)

            self.data.append([])
            for ipt, pt_bin in enumerate(PT_BIN_LIST[icent]):
                if isMC:
                    data = mcDataInterval(df, pre_cut, cent_bin, pt_bin, MASS_BIN, dataType, is_single_matter_type)
                else:
                    data = DataInterval(df, pre_cut, cent_bin, pt_bin, MASS_BIN, dataType, is_single_matter_type)
                if model_list is not None:
                    data.setModel(model_list[icent][ipt])
                self.data[icent].append(deepcopy(data))

    def getBestBDTIndex(self, icent, ipt, isig, ibkg):
        if self.bestBDTIndex is None:
            raise ValueError("Best BDT index not set.")
        return self.bestBDTIndex[icent][ipt][isig][ibkg]
    
    def getDataInterval(self, icent, ipt):
        return self.data[icent][ipt]

    def getDF(self, icent, ipt):
        return self.data[icent][ipt].getDF()
    
    def getPrecutEfficiency(self, icent, ipt):
        if self.precut_efficiency is None:
            raise ValueError("Precut efficiency not set.")
        return self.precut_efficiency[icent][ipt]

    def getSignalCount(self, icent, ipt, isig, ibkg, imodel):
        if self.signalCount is None:
            raise ValueError("Signal count not set.")
        return self.signalCount[icent][ipt][isig][ibkg][imodel]
    
    def getYieldHist(self, icent, isig, ibkg, SystUnc, absorbfactor_list, f_corr, scale_factor=1):
        hStat = ROOT.TH1F("hStat", ";#it{p}_{T} (GeV/c);R", len(self.hist_ptBins[icent]) - 1, self.hist_ptBins[icent])
        hSyst = ROOT.TH1F("hSyst", ";#it{p}_{T} (GeV/c);R", len(self.hist_ptBins[icent]) - 1, self.hist_ptBins[icent])
        print(f"Cent {self.CENT_BIN_LIST[icent][0]}-{self.CENT_BIN_LIST[icent][1]}%")
        for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
            best_index = self.getBestBDTIndex(icent, ipt, isig, ibkg)
            print(f"In {pt_bin[0]} < pT < {pt_bin[1]}, best BDT efficiency is", self.bdt_efficiency[best_index])
            
            # f_precut_corr = Branching Ratio * Event Number
            corrfactor = f_corr * self.precut_efficiency[icent][ipt] * self.bdt_efficiency[best_index] * absorbfactor_list[icent][ipt] * (pt_bin[1] - pt_bin[0])
            if not self.is_single_matter_type:
                corrfactor = corrfactor * 2
            hypyield  = self.signalCount[icent][ipt][isig][ibkg][best_index] / corrfactor
            hypyield_error = self.signalError[icent][ipt][isig][ibkg][best_index] / corrfactor

            hStat.SetBinContent(ipt+1, hypyield)
            hStat.SetBinError(ipt+1, hypyield_error)
            hSyst.SetBinContent(ipt+1, hypyield)
            hSyst.SetBinError(ipt+1, math.sqrt( math.pow( SystUnc[icent][ipt], 2) + math.pow( hypyield * (1 - absorbfactor_list[icent][ipt]) * 0.5, 2) ) )

        hStat.Scale(scale_factor)
        hSyst.Scale(scale_factor)
        return hStat, hSyst
    
    def setModel(self, model_list):
        for icent, cent_bin in enumerate(self.CENT_BIN_LIST):
            for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
                self.data[icent][ipt].setModel(model_list[icent][ipt])

    def setBDTEfficiency(self, bdt_efficiency_list):
        self.bdt_efficiency = bdt_efficiency_list
    
    def setBestBDT(self, exp_significance_list, bdt_efficiency_list):
        self.setBDTEfficiency(bdt_efficiency_list)
        self.bestBDTIndex = utils.createEmptyList( [len(self.CENT_BIN_LIST), len(self.PT_BIN_LIST[0]), len(self.SIGPDF_LIST)] )
        for icent, cent_bin in enumerate(self.CENT_BIN_LIST):
            for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
                for isig, sigfunc in enumerate(self.SIGPDF_LIST):
                    for ibkg, bkgfunc in enumerate(self.BKGPDF_LIST):
                        best_bdt_index = np.argmax(exp_significance_list[icent][ipt][isig][ibkg] * bdt_efficiency_list)
                        self.bestBDTIndex[icent][ipt][isig].append(best_bdt_index)

    def setPreCutEfficiency(self, Efficiency):
        self.precut_efficiency = Efficiency

    def addModelPrediction(self, output_margin=True):
        for icent, cent_bin in enumerate(self.CENT_BIN_LIST):
            for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
                self.data[icent][ipt].addModelPrediction(output_margin)

    def invMFit(self, icent, ipt, sigpdf = "Gauss", bkgpdf = "exp", model_threshold = None,name = None, para = None, isMC = False, ifDrawStats = True, ifDebug = False,outfile = None):
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
    
    def doSimultaneousFits(self, bkg_deuteron, bkg_uncorrelated, MODEL_Eff_LIST, score_eff_arrays_dict,
                           mcpara_list = None, fit_massbin = [2.96, 3.02], outfile = None, fix_bkg_peak = True, ifDebug = False):
        signalCount = utils.createEmptyList( [len(self.CENT_BIN_LIST), len(self.PT_BIN_LIST[0]), len(self.SIGPDF_LIST), len(self.BKGPDF_LIST)] )
        expBkgCount = utils.createEmptyList( [len(self.CENT_BIN_LIST), len(self.PT_BIN_LIST[0]), len(self.SIGPDF_LIST), len(self.BKGPDF_LIST)] )
        signalError = utils.createEmptyList( [len(self.CENT_BIN_LIST), len(self.PT_BIN_LIST[0]), len(self.SIGPDF_LIST), len(self.BKGPDF_LIST)] )
        for icent, cent_bin in enumerate(self.CENT_BIN_LIST):
            outdir_cent = outfile.mkdir(f"Cent{cent_bin[0]}_{cent_bin[1]}")
            for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
                outdir_pt = outdir_cent.mkdir(f"pT{pt_bin[0]}_{pt_bin[1]}")
                binkey = f"pT{pt_bin[0]}_{pt_bin[1]}"
                data_se = self.getDataInterval(icent, ipt)
                bkg_me_deuteron = bkg_deuteron.getDataInterval(icent, ipt)
                bkg_me_uncorrelated = bkg_uncorrelated.getDataInterval(icent, ipt)
                corr_bkg_peak = None
                for isig, sigfunc in enumerate(self.SIGPDF_LIST):
                    outdir_sig = outdir_pt.mkdir(sigfunc)
                    for ibkg, bkgfunc in enumerate(self.BKGPDF_LIST):
                        outdir_bkg = outdir_sig.mkdir(bkgfunc)
                        for imodel, model_eff in enumerate(MODEL_Eff_LIST):
                            if model_eff != 1:
                                model_threshold = score_eff_arrays_dict[binkey][0][imodel]
                            else:
                                model_threshold = None
                            (signal_num, signal_error, exp_bkg_num, bkg_peak_value) = data_se.simultaneousFit(bkg_me_deuteron, bkg_me_uncorrelated, sigfunc, corr_bkgpdf=bkgfunc, uncorr_bkgpdf="pol1", 
                                                                            mcpara=mcpara_list[icent][ipt][0], fit_massbin=fit_massbin, model_eff=model_eff, model_threshold=model_threshold,
                                                                            outfile=outdir_bkg, corr_bkg_peak=corr_bkg_peak, fix_bkg_peak=fix_bkg_peak,ifDebug=ifDebug)
                            ##### fix the peak value of the correlated background for high pt interval #####
                            if ipt == 0 and imodel == len(MODEL_Eff_LIST) - 1:
                                corr_bkg_peak = bkg_peak_value
                            signalCount[icent][ipt][isig][ibkg].append(signal_num)
                            signalError[icent][ipt][isig][ibkg].append(signal_error)
                            expBkgCount[icent][ipt][isig][ibkg].append(exp_bkg_num)
        self.signalCount = signalCount
        self.signalError = signalError
        self.expBkgCount = expBkgCount
        return signalCount, signalError, expBkgCount
    
############### MC Data Group ###############
class mcDataGroup(DataGroup):
    'MC data frame in specific centrality and pt bin'

    def __init__(self, df, pre_cut = None, CENT_BIN_LIST = [[-999, 999]], PT_BIN_LIST = [[[-999, 999]]], MASS_BIN = [2.96, 3.04], SIGPDF_LIST = ["dscb"], BKGPDF_LIST = ["none"], dataType = "", is_single_matter_type = False, model_list = None):
        self._build_data(df, pre_cut, CENT_BIN_LIST, PT_BIN_LIST, MASS_BIN, SIGPDF_LIST, BKGPDF_LIST, dataType, is_single_matter_type=is_single_matter_type, model_list=model_list, isMC=True)

    def calculateEfficiency(self, MCGenPD, pt_spectrum, eventNumber_in_realdata, BR, absorbfactor_list=None, is_single_matter_type=False):
        Efficiency = utils.createEmptyList( [len(self.CENT_BIN_LIST)] )
        ExpCorrectedSignal = utils.createEmptyList( [len(self.CENT_BIN_LIST)] )
        for icent, cent_data in enumerate(self.data):
            for ipt, data_interval in enumerate(cent_data):
                f_absorption = 1.0
                if absorbfactor_list is not None:
                    f_absorption = absorbfactor_list[icent][ipt]
                eff, expsig = data_interval.calculateEfficiency(MCGenPD, pt_spectrum, eventNumber_in_realdata, BR, f_absorption = f_absorption, is_single_matter_type = is_single_matter_type)
                Efficiency[icent].append(eff)
                ExpCorrectedSignal[icent].append(expsig)
        return Efficiency, ExpCorrectedSignal
    
    def getExpSignificance(self, MODEL_Eff_LIST, expBkgCount):
        ExpSignificance = utils.createEmptyList( [len(self.CENT_BIN_LIST), len(self.PT_BIN_LIST[0]), len(self.SIGPDF_LIST), len(self.BKGPDF_LIST)] )
        for icent, cent_bin in enumerate(self.CENT_BIN_LIST):
            for ipt, pt_bin in enumerate(self.PT_BIN_LIST[icent]):
                for isig, sigfunc in enumerate(self.SIGPDF_LIST):
                    for ibkg, bkgfunc in enumerate(self.BKGPDF_LIST):
                        for imodel, modelEff in enumerate(MODEL_Eff_LIST):
                            ExpSignificance[icent][ipt][isig][ibkg].append(self.data[icent][ipt].ExpCorrectedSignal * modelEff / 
                                                                           math.sqrt(self.data[icent][ipt].ExpCorrectedSignal + expBkgCount[icent][ipt][isig][ibkg][imodel]))
        return ExpSignificance

    def saveEfficiencyPlots(self, outfile):
        for icent, cent_data in enumerate(self.data):
            c = utils.TCanvas(f"c_eff_Cent{self.CENT_BIN_LIST[icent][0]}_{self.CENT_BIN_LIST[icent][1]}", "Efficiency vs pT")
            outdir_cent = outfile.Get(f"Cent{self.CENT_BIN_LIST[icent][0]}_{self.CENT_BIN_LIST[icent][1]}")

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

            outdir_cent.cd()
            c.Write()