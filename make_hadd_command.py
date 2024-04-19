from samples.latest import SAMPLE_LIST
from reading.files import get_rootfiles
import os
import argparse

parser = argparse.ArgumentParser(description='Make hadd command for a given sample')
parser.add_argument("sample", type=str)
parser.add_argument("dest", type=str)

args = parser.parse_args()

sample = SAMPLE_LIST.lookup(args.sample)
files = sample.get_files()

print(len(files))
print(files[0])

destdir = args.dest
destfile = os.path.join(destdir, args.sample + ".root")
with open("hadd_command.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("hadd -n 0 -fk %s %s\n" % ( destfile, " ".join(files)))
    f.write("echo 'done'\n")

