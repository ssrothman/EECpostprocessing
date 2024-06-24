#!/bin/bash

python process.py DYJetsToLL_HT-0to70 Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-70to100 Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-100to200 Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-200to400 Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-400to600 Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-600to800 Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-800to1200 Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-1200to2500 Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-2500toInf Beff ak8 inclusive --Zreweight
python process.py DYJetsToLL Beff ak8 inclusive --Zreweight
