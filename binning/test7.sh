for i in {0..5}
do
    echo $i
    python scripts/fill_all_res4_hists.py Apr_23_2025 EECres4tee --total_boot 0 -j8 --statN 6 --statK $i --sample Herwig_inclusive --objsysts --wtsyst --hists directcov_gen directcov_reco reco unmatchedReco untransferedReco gen unmatchedGen untransferedGen
done
