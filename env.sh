source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
unset PYTHONPATH
source $PWD/venv/bin/activate
export PYTHONHOME=/cvmfs/sft.cern.ch/lcg/releases/Python/3.11.9-2924c/x86_64-el9-gcc13-opt
export PATH=$PWD/skimming/scripts:$PWD/binning/scripts:$PATH
export PYTHONPATH=/cvmfs/sft.cern.ch/lcg/releases/torch/2.5.1-f4461/x86_64-el9-gcc13-opt/lib/python3.11/site-packages:$PWD:$PYTHONPATH