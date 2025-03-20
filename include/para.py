import ROOT

CENT_BIN_LIST = [[0, 999]]
# PT_BIN_LIST = [[[2, 2.4], [2.4, 3.5], [3.5, 5]]]
PT_BIN_LIST = [[[2, 3], [3, 5]]]
PT_BIN_LIST_NEW = [[[2, 2.6], [2.6, 5]]]
MASS_BIN = [2.96, 3.04]
MODEL_Eff_LIST = [0.1 + 0.01*x for x in range(0, 91)]
ColorList = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kBlack] 