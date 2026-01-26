runtag = 'May_14_2025'

samples = [
    "herwig_glu",
    "herwig_glu_TeV",
    "herwig_glu_TeV_gg",
    "herwig_glu_TeV_nospin",
    "herwig_glu_TeV_nospin_gg",
    "herwig_glu_gg",
    "herwig_glu_nospin",
    "herwig_glu_nospin_gg",
    "herwig_q",
    "herwig_q_TeV",
    "herwig_q_TeV_nogg",
    "herwig_q_TeV_nospin",
    "herwig_q_TeV_nospin_nogg",
    "herwig_q_nogg",
    "herwig_q_nospin",
    "herwig_q_nospin_nogg",
    "pythia_glu",
    "pythia_glu_TeV",
    "pythia_glu_TeV_nospin",
    "pythia_glu_nospin",
    "pythia_q",
    "pythia_q_TeV",
    "pythia_q_TeV_nospin",
    "pythia_q_nospin"
]

destination = "GENONLY_res4.json"

thedict : dict[str, str | dict] = {
    "base" : ""
}

for sample in samples:
    thedict[sample] = {
        "tag" : "%s_%s" % (sample, runtag),
        "location" : "simon-LPC",
        "era" : "skip",
        "flags" : {},
        "label" : sample.replace("_", ", "),
        "xsec" : 1.0,
        "color" : ""
    }

import json
with open(destination, 'w') as f:
    json.dump({runtag : thedict}, f, indent=4)