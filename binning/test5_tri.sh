for i in {0..5}
do
    echo $i
    python scripts/fill_all_res4_hists.py Apr_23_2025 EECres4triangle --total_boot 0 -j1 --statN 6 --statK $i --objsysts --wtsysts --hists directcov_reco directcov_gen --slurm
done
