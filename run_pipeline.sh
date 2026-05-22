#!/bin/bash
set -e

WORKSPACE_BASE="/eos/user/d/dponman"
CFG="unfolding/configs/pythia_proj.json"

echo "=== Combining Pythia HT bins ==="
for table in proj_totalReco proj_totalGen proj_unmatchedReco proj_unmatchedGen proj_transfer; do
    python -m binning.scripts.combine_ht_stack --table $table --output-objsyst nominal
done
python -m binning.scripts.combine_ht_stack --table proj_totalReco --cov --output-objsyst nominal
python -m binning.scripts.combine_ht_stack --table proj_totalGen  --cov --output-objsyst nominal

echo "=== Setting up Pythia workspace ==="
mkdir -p $WORKSPACE_BASE/proj_unfold_workspace
cp $CFG $WORKSPACE_BASE/proj_unfold_workspace/config.json
cd $WORKSPACE_BASE/proj_unfold_workspace
python -m unfolding.scripts.setup_unfolding_workspace
cd -

echo "=== Running Pythia minimization ==="
python -m unfolding.scripts.run_unfolding \
    --reco-path $WORKSPACE_BASE/proj_unfold_workspace/reco_reco \
    --baseline-path $WORKSPACE_BASE/proj_unfold_workspace/mcgen \
    --model-path $WORKSPACE_BASE/proj_unfold_workspace/model \
    --output-dir $WORKSPACE_BASE/proj_unfold_workspace/minimization \
    --negative-penalty 1e6

echo "=== Building Pythia unfolded histogram ==="
python -m unfolding.scripts.build_unfolded_hist \
    --minimization-dir $WORKSPACE_BASE/proj_unfold_workspace/minimization \
    --unfolded-dir $WORKSPACE_BASE/proj_unfold_workspace/unfolded

echo "=== Setting up Herwig workspace ==="
python -m unfolding.setup_herwig_workspace

echo "=== Running Herwig minimization ==="
python -m unfolding.scripts.run_unfolding \
    --reco-path $WORKSPACE_BASE/proj_unfold_workspace_herwig/reco \
    --baseline-path $WORKSPACE_BASE/proj_unfold_workspace_herwig/mcgen \
    --model-path $WORKSPACE_BASE/proj_unfold_workspace_herwig/model \
    --output-dir $WORKSPACE_BASE/proj_unfold_workspace_herwig/minimization \
    --negative-penalty 1e6

echo "=== Building Herwig unfolded histogram ==="
python -m unfolding.scripts.build_unfolded_hist \
    --minimization-dir $WORKSPACE_BASE/proj_unfold_workspace_herwig/minimization \
    --unfolded-dir $WORKSPACE_BASE/proj_unfold_workspace_herwig/unfolded

echo "=== Setting up Data workspace ==="
python -m unfolding.setup_data_workspace

echo "=== Running Data minimization ==="
python -m unfolding.scripts.run_unfolding \
    --reco-path $WORKSPACE_BASE/proj_unfold_workspace_data/reco \
    --baseline-path $WORKSPACE_BASE/proj_unfold_workspace_data/mcgen \
    --model-path $WORKSPACE_BASE/proj_unfold_workspace_data/model \
    --output-dir $WORKSPACE_BASE/proj_unfold_workspace_data/minimization \
    --negative-penalty 1e6

echo "=== Building Data unfolded histogram ==="
python -m unfolding.scripts.build_unfolded_hist \
    --minimization-dir $WORKSPACE_BASE/proj_unfold_workspace_data/minimization \
    --unfolded-dir $WORKSPACE_BASE/proj_unfold_workspace_data/unfolded

echo "=== Plotting comparison ==="
python -m unfolding.scripts.plot_proj_comparison \
    $WORKSPACE_BASE/proj_unfold_workspace/unfolded \
    $WORKSPACE_BASE/proj_unfold_workspace_herwig/unfolded \
    $WORKSPACE_BASE/proj_unfold_workspace_data/unfolded \
    --labels Pythia Herwig Data \
    --output-folder ./plots/comparison

echo "=== Pipeline complete ==="
