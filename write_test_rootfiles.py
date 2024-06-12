import uproot
import pickle
import plotting.EECutil
from samples.latest import SAMPLE_LIST

MCH = SAMPLE_LIST.get_hist('DYJetsToLL', 'EEC', ['tight'])
print(MCH.keys())

reco = MCH['Hreco'].project("dRbin")
covreco = MCH['HcovReco'].project("dRbin_1", "dRbin_2")
recopure = MCH['HrecoPure'].project("dRbin")
covrecopure = MCH['HcovRecoPure'].project("dRbin_1", "dRbin_2")
recobkg = reco - recopure

gen = MCH["Hgen"].project("dRbin")
covgen = MCH["HcovGen"].project("dRbin_1", "dRbin_2")
genpure = MCH['HgenPure'].project("dRbin")
covgenpure = MCH['HcovGenPure'].project("dRbin_1", "dRbin_2")
genbkg = gen - genpure

transfer = MCH['Htrans'].project("dRbin_Reco", "dRbin_Gen")

with uproot.recreate("testdata.root") as f:
    f['reco_total'] = reco.to_numpy(flow=True)
    f['reco_total_cov'] = covreco.to_numpy(flow=True)
    f['reco_unmatched'] = recobkg.to_numpy(flow=True)
    f['reco_pure'] = recopure.to_numpy(flow=True)
    f['reco_pure_cov'] = covrecopure.to_numpy(flow=True)

    f['gen_total'] = gen.to_numpy(flow=True)
    f['gen_total_cov'] = covgen.to_numpy(flow=True)
    f['gen_unmatched'] = genbkg.to_numpy(flow=True)
    f['gen_pure'] = genpure.to_numpy(flow=True)
    f['gen_pure_cov'] = covgenpure.to_numpy(flow=True)
    
    f['transfer'] = transfer.to_numpy(flow=True)
