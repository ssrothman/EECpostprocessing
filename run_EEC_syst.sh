#!/usr/bin/env bash

#SBATCH -J dask-worker
#SBATCH -e run_EEC_syst.err
#SBATCH -o run_EEC_syst.out
#SBATCH -p submit,submit-centos07,submit-gpu-centos07,submit-gpu-a30-centos07
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH -t 100:00:00

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --JER --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --JER --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --JES --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --JES --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_prefire --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_prefire --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_idsf --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_idsf --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_isosf --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_isosf --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_triggersf --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_triggersf --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_scale --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_scale --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_ISR --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_ISR --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_FSR --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_FSR --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_PDFaS --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_PDFaS --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_PU --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_PU --DN

python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_btagSF --UP
python process.py DYJetsToLL EEC ak8 inclusive --treatAsData --wt_btagSF --DN
