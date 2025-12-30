for i in {0..5}
do
    echo $i
    python scripts/fill_all_res4_hists.py Apr_23_2025 EECres4dipole --total_boot 0 -j8 --statN 6 --statK $i 
done
