from coffea.lookup_tools import extractor

ex = extractor()

files = [
'Summer19UL18_V5_MC_L1FastJet_AK8PFPuppi.jec.txt',
'Summer19UL18_V5_MC_L2L3Residual_AK8PFPuppi.jec.txt',
'Summer19UL18_V5_MC_L2Relative_AK8PFPuppi.jec.txt',
'Summer19UL18_V5_MC_L2Residual_AK8PFPuppi.jec.txt',
'Summer19UL18_V5_MC_L3Absolute_AK8PFPuppi.jec.txt',
'Summer19UL18_V5_MC_Uncertainty_AK8PFPuppi.junc.txt',
'Summer19UL18_V5_MC_UncertaintySources_AK8PFPuppi.junc.txt',
]

ex.add_weight_sets(["* * %s"%f for f in files])

ex.finalize()

evaluator = ex.make_evaluator()
