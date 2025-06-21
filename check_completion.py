import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Era", type=str)
parser.add_argument("Sample", type=str)
parser.add_argument("Binner", type=str)
parser.add_argument("ObjSyst", type=str)
parser.add_argument('--basepath', type=str, default='/ceph/submit/data/group/cms/store/user/srothman/EEC')
args = parser.parse_args()

import os
thepath = os.path.join(args.basepath, args.Era, args.Sample, args.Binner)

#print(thepath)
subpaths = os.scandir(thepath)
options = []
for subpath in subpaths:
    if not subpath.is_dir():
        continue
    
    options += [subpath.name]


if len(options) != 1:
    print("ERROR: SHOULD ONLY BE ONE SUBPATH")
    import sys
    sys.exit(1)

the_actual_path = os.path.join(thepath, options[0], args.ObjSyst)

import re
search = re.search("file(\d+)to(\d+)", options[0])
start = int(search.group(1))
end = int(search.group(2))
N = end-start
#print("Expecting %d files"%N)

for subpath in os.scandir(the_actual_path):
    if not subpath.is_dir():
        continue
    scan = os.scandir(os.path.join(the_actual_path, subpath.name))
    filtered = filter(lambda name : name.name.endswith('.parquet'), scan)

    Npq = len(list(filtered))
    if Npq != N:
        print("ERROR: MISMATCH FOR %s"%os.path.join(the_actual_path, subpath.name))
        print("\texpected %d files"%N)
        print("\tobserved %d files"%Npq)
        import sys
        sys.exit(1)
