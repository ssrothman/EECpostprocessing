#!/bin/bash

era="Apr_23_2025"
what="transfer"
nboot=500

#for dset in "Pythia_HT-0to70" "Pythia_HT-70to100" "Pythia_HT-100to200" "Pythia_HT-200to400" "Pythia_HT-400to600" "Pythia_HT-600to800" "Pythia_HT-800to1200" "Pythia_HT-1200to2500" "Pythia_HT-2500toInf"
for dset in "Pythia_HT-400to600"
do
    python fillEECRes4Hist.py $era $dset EECres4tee $what nominal nominal --nboot $nboot
done
