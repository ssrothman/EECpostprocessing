for sample in "Pythia_inclusive" 
do
    for what in "transfer"
    do
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0  --slurm --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0  --statN 2 --statK 0 --slurm --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0  --statN 2 --statK 1 --slurm --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 

        for rng in $(seq 1 100)
        do
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 20 --skipNominal --slurm --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 20 --skipNominal --statN 2 --statK 0 --slurm --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 20 --skipNominal --statN 2 --statK 1 --slurm --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
        done

        for wtsyst in "idsfUp" "idsfDown" "aSUp" "aSDown" "isosfUp" "isosfDown" "triggersfUp" "triggersfDown" "prefireUp" "prefireDown" "PDFaSUp" "PDFaSDown" "scaleUp" "scaleDown" "PUUp" "PUDown" "PDFUp" "PDFDown" "ISRUp" "ISRDown" "FSRUp" "FSRDown"
        do
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --statN 2 --statK 0 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --statN 2 --statK 1 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
        done

        for objsyst in "JES_UP" "JES_DN" "JER_UP" "JER_DN" "UNCLUSTERED_UP" "UNCLUSTERED_DN" "TRK_EFF" "CH_UP" "CH_DN"
        do
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --statN 2 --statK 0 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --statN 2 --statK 1 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" 
        done
    done
done

for sample in "Herwig_inclusive"
do
    for what in "reco" "unmatchedReco" "untransferedReco" "gen" "unmatchedGen" "untransferedGen" "transfer"
    do
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0 --slurm --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0 --statN 2 --statK 0 --slurm --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0 --statN 2 --statK 1 --slurm --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"

        for rng in $(seq 1 100)
        do
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 20 --skipNominal --slurm --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 20 --skipNominal --statN 2 --statK 0 --slurm --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 20 --skipNominal --statN 2 --statK 1 --slurm --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
        done

        for wtsyst in "idsfUp" "idsfDown" "aSUp" "aSDown" "isosfUp" "isosfDown" "triggersfUp" "triggersfDown" "prefireUp" "prefireDown" "PDFaSUp" "PDFaSDown" "scaleUp" "scaleDown" "PUUp" "PUDown" "PDFUp" "PDFDown" "ISRUp" "ISRDown" "FSRUp" "FSRDown"
        do
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --statN 2 --statK 0 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --statN 2 --statK 1 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
        done

        for objsyst in "JES_UP" "JES_DN" "JER_UP" "JER_DN" "UNCLUSTERED_UP" "UNCLUSTERED_DN" "TRK_EFF" "CH_UP" "CH_DN"
        do
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --statN 2 --statK 0 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --statN 2 --statK 1 --slurm  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight"
        done
    done
done
