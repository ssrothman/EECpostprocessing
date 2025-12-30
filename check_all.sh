for sample in "DATA_2018A" "DATA_2018B" "DATA_2018C" "DATA_2018D"
do
    for binner in "EECres4dipole" "EECres4triangle"
    do
        python check_completion.py Apr_23_2025 $sample $binner nominal
    done
done

for sample in "Pythia_HT-0to70" "Pythia_HT-100to200" "Pythia_HT-1200to2500" "Pythia_HT-200to400" "Pythia_HT-2500toInf" "Pythia_HT-400to600" "Pythia_HT-600to800" "Pythia_HT-70to100" "Pythia_HT-800to1200" "Herwig_inclusive"
do
    for syst in "nominal" "JES_UP" "JES_DN" "JER_UP" "JER_DN" "UNCLUSTERED_UP" "UNCLUSTERED_DN" "CH_UP" "CH_DN" "TRK_EFF"
    do
        for binner in "EECres4dipole" "EECres4triangle"
        do
            python check_completion.py Apr_23_2025 $sample $binner $syst
        done
    done
done
