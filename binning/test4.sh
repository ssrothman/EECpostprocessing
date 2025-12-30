for i in {0..5}
do
    echo $i
    python scripts/fill_all_res4_hists.py Apr_23_2025 EECres4dipole --total_boot 0 -j1 --statN 6 --statK $i --samples DATA_2018A DATA_2018B DATA_2018C DATA_2018D --hists reco directcov_reco --objsysts --wtsysts --kinreweight_key None --kinreweight_path None --slurm
done
