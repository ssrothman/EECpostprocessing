for i in {0..299}
do
    echo $i
    python scripts/fill_all_res4_hists.py Apr_23_2025 EECres4tee --total_boot 0 -j4 --statN 300 --statK $i --objsysts --wtsysts --hists directcov_reco directcov_gen
done
