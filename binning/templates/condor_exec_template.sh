#!/bin/bash
set -euxo pipefail

echo "STARTED JOB"

xrdcp root://submit55.mit.edu//store/user/srothman/EEC/TARBALLS/myenv.tgz .
xrdcp root://submit55.mit.edu//store/user/srothman/EEC/TARBALLS/kinSF.tgz .
xrdcp root://submit55.mit.edu//store/user/srothman/EEC/TARBALLS/python.tgz .
xrdcp root://submit55.mit.edu//store/user/srothman/EEC/TARBALLS/pyrandom123.tgz .

tar -xzvf myenv.tgz
tar -xzvf kinSF.tgz
tar -xzvf python.tgz
tar -xzvf pyrandom123.tgz

source myenv/bin/activate

pip install --user ./pyrandom123

COMMAND
