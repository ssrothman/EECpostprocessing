#!/bin/bash
set -e

RUNTAG=v6
DATASET=DYJetsToLL_Pythia
LOCATION=dylan-lxplus-eos
CONFIG=EvtMCprojConfig

TABLES="proj_Gen proj_Reco proj_unmatchedGen proj_unmatchedReco proj_transfer"
for TABLE in $TABLES; do
    echo "Binning $TABLE NOM..."
    python binning/scripts/bin.py $RUNTAG $DATASET NOM $TABLE \
        --location $LOCATION --config-suite $CONFIG
done

echo "Computing covariance for proj_Reco NOM..."
python binning/scripts/bin.py $RUNTAG $DATASET NOM proj_Reco \
    --location $LOCATION --config-suite $CONFIG --cov

echo "Done."
