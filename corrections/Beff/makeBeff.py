from samples.latest import SAMPLE_LIST
import numpy as np
import correctionlib.schemav2 as cs

import json
with open("configs/base.json") as f:
    config = json.load(f)

H = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Beff", ['noBtagSF',
                                                             'genXsec'])['Beff']

effs = {}
corrs = {}

flavnames = ["udsg", "c", "b"]

for tag in ["tight", "medium", "loose"]:
    effs[tag] = {}
    corrs[tag] = {}

    Hproj = H.project("genflav", "pt", 'eta', "btag_"+tag)
    print(Hproj)

    values = Hproj.values(flow=True)
    variances = Hproj.variances(flow=True)

    Bpass = values[:, :, :, 1]
    Bfail = values[:, :, :, 0]

    eff = Bpass/(Bpass + Bfail)
    eff = np.nan_to_num(eff, nan=0, posinf=0, neginf=0)

    effs[tag] = eff

    ptedges = Hproj.axes[1].edges.tolist() + [np.inf]
    etaedges = Hproj.axes[2].edges.tolist()
    print(ptedges)
    print(etaedges)
    print(eff[0,:,:].shape)
    print(eff[0,:,:].tolist())
    print(eff[0,:,:].tolist()[0])

    corr = cs.Correction(
        name=tag,
        version=1,
        description="B tagger efficiency for AK4 CHS jets",
        inputs=[
            cs.Variable(name="pt", type="real", 
                        description="CHS jet pT [GeV]"),
            cs.Variable(name="abseta", type="real", 
                        description="CHS jet |eta|"),
            cs.Variable(name='hadronFlavour', type="int", 
                        description="CHS jet hadron flavour"+
                                    " (0: udsg, 4: c, 5: b)")
        ],
        output=cs.Variable(name="efficiency", type="real", description="B tagger pass rate"),
        data = cs.Category(
            nodetype="category",
            input="hadronFlavour",
            content = [
                cs.CategoryItem(
                    key=0,
                    value = cs.MultiBinning(
                        nodetype="multibinning",
                        inputs=["pt", "abseta"],
                        edges=[ptedges, etaedges],
                        content = eff[0, :, :].ravel().tolist(),
                        flow='error'
                    )
                ),
                cs.CategoryItem(
                    key=4,
                    value = cs.MultiBinning(
                        nodetype="multibinning",
                        inputs=["pt", "abseta"],
                        edges=[ptedges, etaedges],
                        content = eff[1, :, :].ravel().tolist(),
                        flow='error'
                    )
                ),
                cs.CategoryItem(
                    key=5,
                    value = cs.MultiBinning(
                        nodetype="multibinning",
                        inputs=["pt", "abseta"],
                        edges=[ptedges, etaedges],
                        content = eff[2, :, :].ravel().tolist(),
                        flow='error'
                    )
                )
            ]
        )
    )

    import rich
    rich.print(corr)

    corrs[tag] = corr

cset = cs.CorrectionSet(
        schema_version=2, 
        corrections=[corrs[tag] for tag in ["tight", "medium", "loose"]]
)

with open("corrections/Beff/Beff.json", "w") as f:
    f.write(cset.json(exclude_unset=True, indent=4))
