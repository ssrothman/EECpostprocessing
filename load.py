from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np

x = NanoEventsFactory.from_root("NANO_NANO.root", schemaclass=NanoAODSchema).events()

matrix = x.GenMatch.matrix
nrows = ak.flatten(x.GenMatchBK.n_rows)
ncols = ak.flatten(x.GenMatchBK.n_cols)
ntot = nrows * ncols

matrix = ak.unflatten(matrix, ntot, axis=-1)
matrix = ak.unflatten(matrix, np.repeat(ncols, nrows), axis=-1)


wts = x.RecoEECWTS.value
dRs = x.RecoEECDRS.value
nDR = x.RecoEECBK.nDR
nWT = x.RecoEECBK.nWts
nRes3 = x.RecoEECBK.nRes3
nRes4 = x.RecoEECBK.nRes4
ncovPxP = x.RecoEECBK.ncovPxP
ncov3x3 = x.RecoEECBK.ncov3x3
ncov3xP = x.RecoEECBK.ncov3xP
ncov4x4 = x.RecoEECBK.ncov4x4
ncov4x3 = x.RecoEECBK.ncov4x3
ncov4xP = x.RecoEECBK.ncov4xP
nOrd = x.RecoEECBK.nOrders

wts = ak.unflatten(wts, ak.flatten(nWT), axis=-1)
dRs = ak.unflatten(dRs, ak.flatten(nDR), axis=-1)

wtct = np.repeat(ak.flatten(ak.num(dRs, axis=-1)), ak.flatten(nOrd, axis=None))
wts = ak.unflatten(wts, wtct, axis=-1)

covPxP = x.RecoEECCOVPxP.value
covPxP = ak.unflatten(covPxP, ak.flatten(ncovPxP), axis=-1)
covPxPct = np.repeat(ak.flatten(nWT), ak.flatten(nWT))
covPxP = ak.unflatten(covPxP, covPxPct, axis=-1)

orders = x.RecoEECWTSBK.order
orders = ak.unflatten(orders, ak.flatten(nOrd), axis=-1)

#for binning cov, the follow tricks will be necessary
covPDR, _ = ak.broadcast_arrays(dRs[:,:,None,:], wts)
covPDR = ak.flatten(covPDR, axis=-1) #can then be broadcasted with cov for binning

#similarly
covPOrd = ak.local_index(orders, axis=-1) #for binning, better to have contiguous indices
covPOrd, _ = ak.broadcast_arrays(covPOrd, wts);
covPOrd = ak.flatten(covPOrd, axis=-1)

#for the resolved
res3 = x.RecoEECRES3
res3 = ak.unflatten(res3, ak.flatten(nRes3), axis=-1)

res4 = x.RecoEECRES4
res4 = ak.unflatten(res4, ak.flatten(nRes4), axis=-1)

#et
