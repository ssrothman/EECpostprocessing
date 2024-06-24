#!/bin/bash

python process.py DYJetsToLL_HT-0to70 Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-70to100 Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-100to200 Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-200to400 Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-400to600 Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-600to800 Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-800to1200 Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-1200to2500 Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL_HT-2500toInf Kin ak8 inclusive --Zreweight
python process.py DYJetsToLL Kin ak8 inclusive --Zreweight
