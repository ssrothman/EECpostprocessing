#for sample in "Pythia_inclusive" "Pythia_HT-0to70" "Pythia_HT-70to100" "Pythia_HT-100to200" "Pythia_HT-200to400" "Pythia_HT-400to600" "Pythia_HT-600to800" "Pythia_HT-800to1200" "Pythia_HT-1200to2500" "Pythia_HT-2500toInf"
for sample in "Pythia_HT-70to100" "Pythia_HT-100to200" "Pythia_HT-200to400" "Pythia_HT-400to600" "Pythia_HT-600to800" "Pythia_HT-800to1200" "Pythia_HT-1200to2500" "Pythia_HT-2500toInf"
do
    for what in "reco" "unmatchedReco" "untransferedReco" "gen" "unmatchedGen" "untransferedGen" "transfer"
    do
        #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0  --statN 2 --statK 0 --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0  --statN 2 --statK 1 --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
        wait

        for rng in $(seq 0 8)
        do
        #    python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 500 --skipNominal --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
        #    python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 500 --skipNominal --statN 2 --statK 0 --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
        #    python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 500 --skipNominal --statN 2 --statK 1 --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
            wait
        done

        for wtsyst in "idsfUp" "idsfDown" "aSUp" "aSDown" "isosfUp" "isosfDown" "triggersfUp" "triggersfDown" "prefireUp" "prefireDown" "PDFaSUp" "PDFaSDown" "scaleUp" "scaleDown" "PUUp" "PUDown" "PDFUp" "PDFDown" "ISRUp" "ISRDown" "FSRUp" "FSRDown"
        do
            #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --statN 2 --statK 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --statN 2 --statK 1  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
            wait
        done

        for objsyst in "JES_UP" "JES_DN" "JER_UP" "JER_DN" "UNCLUSTERED_UP" "UNCLUSTERED_DN" "TRK_EFF" "CH_UP" "CH_DN"
        do
            #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --statN 2 --statK 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --statN 2 --statK 1  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Pythia_Zkinweight" &
            wait
        done
    done
done

for sample in "Herwig_inclusive"
do
    for what in "reco" "unmatchedReco" "untransferedReco" "gen" "unmatchedGen" "untransferedGen" "transfer"
    do
        #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0 --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0 --statN 2 --statK 0 --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
        python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 0 --statN 2 --statK 1 --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
        wait

        for rng in $(seq 0 8)
        do
            #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 500 --skipNominal --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 500 --skipNominal --statN 2 --statK 0 --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal nominal --nboot 500 --skipNominal --statN 2 --statK 1 --rng $rng --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            wait
        done

        for wtsyst in "idsfUp" "idsfDown" "aSUp" "aSDown" "isosfUp" "isosfDown" "triggersfUp" "triggersfDown" "prefireUp" "prefireDown" "PDFaSUp" "PDFaSDown" "scaleUp" "scaleDown" "PUUp" "PUDown" "PDFUp" "PDFDown" 
        do
            #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --statN 2 --statK 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what nominal $wtsyst --nboot 0 --statN 2 --statK 1  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            wait
        done

        for objsyst in "JES_UP" "JES_DN" "JER_UP" "JER_DN" "UNCLUSTERED_UP" "UNCLUSTERED_DN" "TRK_EFF" "CH_UP" "CH_DN"
        do
            #python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --statN 2 --statK 0  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            python fillEECRes4Hist.py Apr_23_2025 $sample EECres4tee $what $objsyst nominal --nboot 0 --statN 2 --statK 1  --kinreweight_path /home/submit/srothman/work/EEC/plotting/kinSF/Zkin.json --kinreweight_key "Herwig_Zkinweight" &
            wait
        done
    done
done
