from coffea.nanoevents import NanoEventsFactory
from reading.files import get_rootfiles

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

hostid = 'cmseos.fnal.gov'

files_AOD = get_rootfiles(hostid, '/store/group/lpcpfnano/srothman/Nov07_2023_jets_AOD')
AOD = NanoEventsFactory.from_root(files_AOD[0]).events()

files_miniAOD = get_rootfiles(hostid, '/store/group/lpcpfnano/srothman/Nov07_2023_jets_miniAOD')
miniAOD = NanoEventsFactory.from_root(files_miniAOD[0]).events()

jetpt_AOD = ak.flatten(AOD.SimonJetsBK.jetPt)
jetpt_miniAOD = ak.flatten(miniAOD.SimonJetsBK.jetPt)
jetpt_miniAOD_JTB = ak.flatten(miniAOD.selectedPatJetsAK4PFPuppi.pt)
jetpt_cmssw = ak.flatten(miniAOD.TestFullJetTable.pt)
jetpt_AOD = jetpt_AOD[jetpt_AOD > 30]
jetpt_miniAOD = jetpt_miniAOD[jetpt_miniAOD > 30]
jetpt_miniAOD_JTB = jetpt_miniAOD_JTB[jetpt_miniAOD_JTB > 30]
jetpt_cmssw = jetpt_cmssw[jetpt_cmssw > 30]
'''

nPart_AOD = ak.flatten(AOD.SimonJetsBK.nPart)
nPart_miniAOD = ak.flatten(miniAOD.SimonJetsBK.nPart)

nMU_AOD = ak.flatten(AOD.SimonJetsBK.nMU)
nMU_miniAOD = ak.flatten(miniAOD.SimonJetsBK.nMU)

nELE_AOD = ak.flatten(AOD.SimonJetsBK.nELE)
nELE_miniAOD = ak.flatten(miniAOD.SimonJetsBK.nELE)

nHADCH_AOD = ak.flatten(AOD.SimonJetsBK.nHADCH)
nHADCH_miniAOD = ak.flatten(miniAOD.SimonJetsBK.nHADCH)

nEM0_AOD = ak.flatten(AOD.SimonJetsBK.nEM0)
nEM0_miniAOD = ak.flatten(miniAOD.SimonJetsBK.nEM0)

nHAD0_AOD = ak.flatten(AOD.SimonJetsBK.nHAD0)
nHAD0_miniAOD = ak.flatten(miniAOD.SimonJetsBK.nHAD0)

vx_AOD = ak.flatten(AOD.SimonJets.vtx_x)
vx_miniAOD = ak.flatten(miniAOD.SimonJets.vtx_x)

vy_AOD = ak.flatten(AOD.SimonJets.vtx_y)
vy_miniAOD = ak.flatten(miniAOD.SimonJets.vtx_y)

vz_AOD = ak.flatten(AOD.SimonJets.vtx_z)
vz_miniAOD = ak.flatten(miniAOD.SimonJets.vtx_z)

#fromPV_AOD = ak.flatten(AOD.SimonJets.fromPV)
fromPV_miniAOD = ak.flatten(miniAOD.SimonJets.fromPV)

plt.hist([jetpt_AOD, jetpt_miniAOD, jetpt_miniAOD_JTB, jetpt_cmssw], bins=50, histtype='step', label=['AOD', 'miniAOD', 'JTB', 'cms'], density=True)
plt.legend()
plt.show()
plt.title("miniAOD fromPV")
plt.hist(fromPV_miniAOD, bins=np.arange(5))
plt.show()

plt.title("Particle multiplicity")
maxN = ak.max([nPart_AOD, nPart_miniAOD])
plt.hist([nPart_AOD, nPart_miniAOD], bins=np.arange(0, maxN, 1), histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.xlabel("Particles per jet")
plt.ylabel("Events [density]")
plt.legend()
plt.savefig("nPart.png")
plt.show()

plt.title("vertex x")
plt.hist([vx_AOD, vx_miniAOD], bins=100, histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.yscale("log")
plt.xlabel("Vertex x coordinate [cm(?)]")
plt.ylabel("Events [density]")
plt.legend()
plt.savefig("vx.png")
plt.show()

plt.title("vertex y")
plt.hist([vy_AOD, vy_miniAOD], bins=100, histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.legend()
plt.yscale("log")
plt.xlabel("Vertex y coordinate [cm(?)]")
plt.ylabel("Events [density]")
plt.savefig("vy.png")
plt.show()

plt.title("vertex z")
plt.hist([vz_AOD, vz_miniAOD], bins=100, histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.yscale("log")
plt.xlabel("Vertex z coordinate [cm(?)]")
plt.ylabel("Events [density]")
plt.legend()
plt.savefig("vz.png")
plt.show()

plt.title("Muons")
plt.ylabel("Events [density]")
plt.xlabel("Muons per jet")
bins = ak.max([nMU_AOD, nMU_miniAOD]) - ak.min([nMU_AOD, nMU_miniAOD])
plt.hist([nMU_AOD, nMU_miniAOD], bins=bins, histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.legend()
plt.show()

plt.title("Electrons")
plt.ylabel("Events [density]")
plt.xlabel("Electrons per jet")
bins = ak.max([nELE_AOD, nELE_miniAOD]) - ak.min([nELE_AOD, nELE_miniAOD])
plt.hist([nELE_AOD, nELE_miniAOD], bins=bins, histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.legend()
plt.show()

plt.title("Charged hadrons")
plt.ylabel("Events [density]")
plt.xlabel("HADCH per jet")
bins = ak.max([nHADCH_AOD, nHADCH_miniAOD]) - ak.min([nHADCH_AOD, nHADCH_miniAOD])
plt.hist([nHADCH_AOD, nHADCH_miniAOD], bins=bins, histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.legend()
plt.show()

plt.title("HAD0")
plt.ylabel("Events [density]")
plt.xlabel("HAD0 per jet")
bins = ak.max([nHAD0_AOD, nHAD0_miniAOD]) - ak.min([nHAD0_AOD, nHAD0_miniAOD])
plt.hist([nHAD0_AOD, nHAD0_miniAOD], bins=bins, histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.legend()
plt.show()

plt.title("EM0")
plt.ylabel("Events [density]")
plt.xlabel("EM0 per jet")
bins = ak.max([nEM0_AOD, nEM0_miniAOD]) - ak.min([nEM0_AOD, nEM0_miniAOD])
plt.hist([nEM0_AOD, nEM0_miniAOD], bins=bins, histtype='step', label=['AOD', 'miniAOD'], density=True)
plt.legend()
plt.show()

'''
