for binner in "EECres4tee" "EECres4dipole" "EECres4triangle"
do
    #for sample in "Pythia_HT-0to70" "Pythia_HT-70to100" "Pythia_HT-100to200" "Pythia_HT-200to400" "Pythia_HT-400to600" "Pythia_HT-600to800" "Pythia_HT-800to1200" "Pythia_HT-1200to2500" "Pythia_HT-2500toInf" "Pythia_inclusive" "Herwig_inclusive" 
    for sample in "Pythia_HT-70to100" 
    do
        for syst in "nominal" "JES_UP" "JES_DN" "JER_UP" "JER_DN" "UNCLUSTERED_UP" "UNCLUSTERED_DN" "CH_UP" "CH_DN" "TRK_EFF"
        do
            echo $sample $binner $syst
            python process.py $sample $binner MC --samplelist latest_res4 --verbose 0 --syst $syst
        done
    done

    for sample in "DATA_2018A" "DATA_2018B" "DATA_2018C" "DATA_2018D"
    do
        echo $sample $binner nominal
        python process.py $sample $binner data --samplelist latest_res4 --verbose 0  --syst nominal
    done

    #for sample in "ST_t" "ST_tW" "ST_tW_anti" "ST_t_anti" "TT" "WW" "WZ" "ZZ"
    #do
    #    echo $sample $binner nominal
    #    python process.py $sample $binner MC --samplelist latest_res4 --verbose 0  --syst nominal
    #done
done
