#!/bin/bash

set -euxo pipefail # Exit with nonzero exit code if anything fails 

N=$1

tar -xf postprocessing.tgz

mv CONDOR_EXEC.py postprocessing/
mv args.json postprocessing/
mv filelist.json postprocessing/
cd postprocessing/

python CONDOR_EXEC.py $1

(r=3;while ! xrdcp ./result.pkl $2 ; do ((--r))||exit;sleep 60;done)

rm -f ./input.root

cd ../
rm -rf postprocessing/
