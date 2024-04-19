from samples.latest import SAMPLE_LIST
import numpy as np

import json
with open("configs/base.json") as f:
    config = json.load(f)

data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", [])
MC = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", [])
WW = SAMPLE_LIST.lookup("WW").get_hist("Kin", [])
ZZ = SAMPLE_LIST.lookup("ZZ").get_hist("Kin", [])
WZ = SAMPLE_LIST.lookup("WZ").get_hist("Kin", [])
TT = SAMPLE_LIST.lookup("TTTo2L2Nu").get_hist("Kin", [])

Hdata = data['HZ'].project("Zpt")
HMC = MC['HZ'].project("Zpt")
HWW = WW['HZ'].project("Zpt")
HZZ = ZZ['HZ'].project("Zpt")
HWZ = WZ['HZ'].project("Zpt")
HTT = TT['HZ'].project("Zpt")

HMC = HMC * config['totalLumi'] * config['xsecs']['DYJetsToLL'] / MC['sumwt'] * 1000
HWW = HWW * config['totalLumi'] * config['xsecs']['WW'] / WW['sumwt'] * 1000
HZZ = HZZ * config['totalLumi'] * config['xsecs']['ZZ'] / ZZ['sumwt'] * 1000
HWZ = HWZ * config['totalLumi'] * config['xsecs']['WZ'] / WZ['sumwt'] * 1000
HTT = HTT * config['totalLumi'] * config['xsecs']['TTTo2L2Nu'] / TT['sumwt'] * 1000

#HMCTotal = HMC + HWW + HZZ + HWZ + HTT
HMCTotal = HMC

ratio = Hdata.values(flow=False)/HMCTotal.values(flow=False)
print(ratio)

ratio = np.where(ratio >2, 2, ratio)
ratio = np.where(ratio <0, 0, ratio)
ratio = np.where(np.isnan(ratio), 1, ratio)

import matplotlib.pyplot as plt
import numpy as np

plt.scatter(np.arange(len(ratio)), ratio)

import correctionlib.schemav2 as cs

corr = cs.Correction(
    name="Zwt",
    version=1,
    description="Z kinematics data/MC weight",
    inputs=[
        cs.Variable(name="Zpt", type="real", description="Z pT [GeV]"),
    ],
    output=cs.Variable(name="weight", type="real", description="Z kinematics data/MC weight"),
    data = cs.Binning(
        nodetype="binning",
        input="Zpt",
        edges = Hdata.axes[0].edges.tolist(),
        content = ratio.tolist(),
        flow='clamp'
    )
)

import rich
rich.print(corr)

cset = cs.CorrectionSet(
        schema_version=2, 
        corrections=[corr]
)
with open("corrections/Zkin/Zwt.json", "w") as f:
    f.write(cset.json(exclude_unset=True, indent=4))

plt.show()
