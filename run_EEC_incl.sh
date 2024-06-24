#!/bin/bash

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData
python process.py TTTo2L2Nu EEC ak8 inclusive --treatAsData
python process.py ZZ EEC ak8 inclusive --treatAsData
python process.py WW EEC ak8 inclusive --treatAsData
python process.py WZ EEC ak8 inclusive --treatAsData
python process.py ST_tW_top EEC ak8 inclusive --treatAsData
python process.py ST_tW_antitop EEC ak8 inclusive --treatAsData
