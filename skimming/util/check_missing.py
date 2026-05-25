from asyncio import subprocess
import os.path
import os
import json
import fcntl
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Sequence
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from skimming.objects.AllObjects import AllObjects
from general.fslookup.location_lookup import location_lookup, lookup_hostid
from skimming.util.parse_workspace import has_existing_submission_dir, infer_workspace_metadata, is_workspace

# Central cache configuration
CACHE_DIR = os.path.expanduser("~/.cache/eec_skimming")
CACHE_FILE = os.path.join(CACHE_DIR, "uniqueid_cache.json")

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_cache():
    ensure_cache_dir()
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                #fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    return {}
                finally:
                    pass
                    #fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_cache(cache):
    ensure_cache_dir()
    # Merge with current on-disk state under a single exclusive lock
    # to avoid clobbering entries added by concurrent processes.
    with open(CACHE_FILE, 'a+') as f:
        #fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            current_cache = {}
            existing_content = f.read()
            if existing_content.strip() != "":
                try:
                    parsed = json.loads(existing_content)
                    if isinstance(parsed, dict):
                        current_cache = parsed
                except json.JSONDecodeError:
                    current_cache = {}

            merged_cache = dict(current_cache)
            merged_cache.update(cache)

            f.seek(0)
            f.truncate()
            json.dump(merged_cache, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        finally:
            pass
            #fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def get_expected_name(target_file, hostid, uniqueid_cache):
    # Check cache first
    if target_file in uniqueid_cache:
        cached_entry = uniqueid_cache[target_file]
        if 'uniqueid' in cached_entry:
            # Use cached value
            return target_file, cached_entry['uniqueid'], None
    
    # Cache miss: read ROOT file and extract uniqueid
    events = NanoEventsFactory.from_root(
        'root://' + hostid + '//' + target_file + ":Events", 
        schemaclass=NanoAODSchema,
        mode='virtual'
    ).events()
    uniqueid = AllObjects.get_uniqueid(events) # type: ignore
    # Return the computed uniqueid for cache update
    return target_file, uniqueid, uniqueid

def check_one_table(target_files, skimfs, hostid, skimbase, 
                    configsuite, runtag, dataset, objsyst, table, 
                    dont_short_circuit, j,
                    uniqueid_cache) -> tuple[set[str], set[str]]:
    skimpath = os.path.join(
        skimbase, 
        configsuite,
        runtag,
        dataset,
        objsyst,
        table
    )

    listdir = skimfs.listdir(skimpath)

    skimresults = set()
    for item in listdir:
        if not (item['name'].endswith('.parquet') or item['name'].endswith('.json')):
            continue
        if item['name'] == 'merged.json':
            continue
        if item['size'] == 0:
            print("[WARNING] Found empty skim result file %s, skipping." % item['name'])
            continue
        name = os.path.basename(item['name'])
        # strip .parquet or .json extension to get the expected target file name
        name = name.replace('.parquet', '').replace('.json', '')
        skimresults.add(name)

    print("[INFO] Found %d skim results for table %s." % (len(skimresults), table))
    print("[INFO] Found %d target files." % len(target_files))

    if not dont_short_circuit and len(target_files) == len(skimresults):
        print("[INFO] Number of skim results matches number of target files, assuming no missing files for table %s." % table)
        return set(), set()

    missing_files = set()

    max_workers = max(1, j)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_target_file = {executor.submit(get_expected_name, target_file, hostid, uniqueid_cache): target_file for target_file in target_files}
        for future in tqdm(as_completed(future_to_target_file), total=len(target_files)):
            target_file, expected_name, computed_uniqueid = future.result()
            
            # Update cache if we computed a new uniqueid
            if computed_uniqueid is not None:
                uniqueid_cache[target_file] = {'uniqueid': computed_uniqueid}
            
            if expected_name in skimresults:
                # success!
                skimresults.remove(expected_name) # remove from the set of skim results so that 
                                                # at the end, any remaining skim results in the set 
                                                # are unexpected/skimming errors
            else:
                missing_files.add(target_file)

    return missing_files, skimresults

def check_workspace(workspace_path : str, dont_short_circuit : bool, j : int):    
    with open(os.path.join(workspace_path, 'target_files.txt'), 'r') as f:
        target_files = [line.strip() for line in f if line.strip()]

    metadata, tables = infer_workspace_metadata(workspace_path)

    skimfs, skimbase = location_lookup(metadata['location'])
    hostid = lookup_hostid(metadata['input_location'])

    # Load cache before processing
    uniqueid_cache = load_cache()
    print("[INFO] Uniqueid cache loaded: %d entries from %s" % (len(uniqueid_cache), CACHE_FILE))

    missing_files : set[str] = set()
    skimresults : set[str] = set()

    for table in tables:
        print("[INFO] Checking table %s..." % table)
        missing, skimres = check_one_table(
            target_files, skimfs, hostid, skimbase,
            metadata['config_suite'], metadata['runtag'], 
            metadata['dataset'], metadata['objsyst'], table,
            dont_short_circuit, j,
            uniqueid_cache
        )

        missing_files.update(missing)
        skimresults.update(skimres)

    # Save updated cache with lock
    save_cache(uniqueid_cache)
    print("[INFO] Uniqueid cache saved with %d entries" % len(uniqueid_cache))

    if len(skimresults) > 0:
        print("[WARNING] Found %d skim result files that do not correspond to any target file:" % len(skimresults))
        for s in sorted(skimresults):
            print("  ", s)

    return missing_files, skimresults

def choose_next_suffix(workspace: str, scheduler: str) -> int:
    """Choose next suffix for missing files, checking for existing artifacts from the given scheduler."""
    idx = 1
    while True:
        target_files_missing = os.path.join(workspace, f"target_files_missing_{idx}.txt")
        skimscript_missing = os.path.join(workspace, f"skimscript_missing_{idx}.py")
        submit_missing = os.path.join(workspace, f"submit_{scheduler}_missing_{idx}")
        # Also check for alternative extension (.sh for slurm/local, .sub for condor)
        ext = ".sub" if scheduler == "condor" else ".sh"
        submit_missing_ext = os.path.join(workspace, f"submit_{scheduler}_missing_{idx}{ext}")
        condor_exec_missing = os.path.join(workspace, f"condor_exec_missing_{idx}.sh")
        if not any(
            os.path.exists(p)
            for p in [target_files_missing, skimscript_missing, submit_missing, submit_missing_ext, condor_exec_missing]
        ):
            return idx
        idx += 1

def make_missing_skimscript(workspace: str, suffix: int) -> str:
    src = os.path.join(workspace, "skimscript.py")
    if not os.path.exists(src):
        raise RuntimeError(f"Workspace is missing required file: {src}")

    dst_name = f"skimscript_missing_{suffix}.py"
    dst = os.path.join(workspace, dst_name)

    with open(src, "r", encoding="utf-8") as f:
        content = f.read()

    replaced = content.replace("./target_files.txt", f"./target_files_missing_{suffix}.txt")
    replaced = replaced.replace("target_files.txt", f"target_files_missing_{suffix}.txt")

    with open(dst, "w", encoding="utf-8") as f:
        f.write(replaced)

    return dst_name

def make_missing_target_file(workspace: str, suffix: int, missing_files: Sequence[str]) -> str:
    dst_name = f"target_files_missing_{suffix}.txt"
    dst = os.path.join(workspace, dst_name)
    with open(dst, "w", encoding="utf-8") as f:
        for tf in sorted(missing_files):
            f.write(tf + "\n")
    return dst_name

def make_missing_slurm_submit(
    workspace: str,
    name: str,
    suffix: int,
    files_per_job: int,
    nfiles: int,
    skimscript_name: str,
    mem: str,
) -> str:
    """Create SLURM submit script for missing files."""
    template = os.path.join(os.path.dirname(__file__), "..", "scaleout", "templates", "slurm_template.sh")
    template = os.path.abspath(template)
    if not os.path.exists(template):
        raise RuntimeError(f"Missing slurm template file: {template}")

    with open(template, "r", encoding="utf-8") as f:
        content = f.read()

    njobs = nfiles // files_per_job
    content = content.replace("NAME", name)
    content = content.replace("NJOBS", str(njobs))
    content = content.replace("MEM", mem)
    content = content.replace("WORKINGDIR", workspace)
    content = content.replace("FILES_PER_JOB", str(files_per_job))
    content = content.replace("NFILES", str(nfiles))
    content = content.replace("python skimscript.py $index", f"python {skimscript_name} $index")

    submit_name = f"submit_slurm_missing_{suffix}.sh"
    submit_path = os.path.join(workspace, submit_name)
    with open(submit_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Mirror behavior of template scripts by marking executable.
    st = os.stat(submit_path)
    os.chmod(submit_path, st.st_mode | 0o111)

    return submit_name


def make_missing_condor_submit(
    workspace: str,
    name: str,
    suffix: int,
    files_per_job: int,
    nfiles: int,
    skimscript_name: str,
    mem: str,
) -> tuple[str, str]:
    """Create Condor submit files for missing files.
    
    Returns tuple of (submit_file_name, exec_file_name).
    """
    submit_template = os.path.join(os.path.dirname(__file__), "..", "scaleout", "templates", "condor_submit_template.sh")
    exec_template = os.path.join(os.path.dirname(__file__), "..", "scaleout", "templates", "condor_exec_template.sh")
    submit_template = os.path.abspath(submit_template)
    exec_template = os.path.abspath(exec_template)
    
    if not os.path.exists(submit_template):
        raise RuntimeError(f"Missing condor submit template file: {submit_template}")
    if not os.path.exists(exec_template):
        raise RuntimeError(f"Missing condor exec template file: {exec_template}")

    target_name = f"target_files_missing_{suffix}.txt"
    exec_name = f"condor_exec_missing_{suffix}.sh"

    # Create submit file
    with open(submit_template, "r", encoding="utf-8") as f:
        submit_content = f.read()

    njobs = nfiles // files_per_job
    submit_content = submit_content.replace("NAME", name)
    submit_content = submit_content.replace("NJOBS", str(njobs))
    submit_content = submit_content.replace("MEM", mem)
    submit_content = submit_content.replace("condor_exec.sh", exec_name)
    submit_content = submit_content.replace("skimscript.py", skimscript_name)
    submit_content = submit_content.replace("target_files.txt", target_name)

    submit_name = f"submit_condor_missing_{suffix}.sub"
    submit_path = os.path.join(workspace, submit_name)
    with open(submit_path, "w", encoding="utf-8") as f:
        f.write(submit_content)

    # Create exec file
    with open(exec_template, "r", encoding="utf-8") as f:
        exec_content = f.read()

    exec_content = exec_content.replace("FILES_PER_JOB", str(files_per_job))
    exec_content = exec_content.replace("NFILES", str(nfiles))

    exec_path = os.path.join(workspace, exec_name)

    exec_content = exec_content.replace("python skimscript.py $index", f"python {skimscript_name} $index")

    with open(exec_path, "w", encoding="utf-8") as f:
        f.write(exec_content)

    # Mark exec file as executable
    st = os.stat(exec_path)
    os.chmod(exec_path, st.st_mode | 0o111)

    return submit_name, exec_name


def make_missing_local_submit(
    workspace: str,
    suffix: int,
    nfiles: int,
    skimscript_name: str,
) -> str:
    """Create a local bash script that runs all missing jobs sequentially."""
    submit_name = f"submit_local_missing_{suffix}.sh"
    submit_path = os.path.join(workspace, submit_name)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"for i in $(seq 0 {nfiles - 1}); do",
        f"    python {skimscript_name} \"$i\" 2>&1 | tee local/skim_missing_{suffix}_$i.log",
        "done",
        "",
    ]

    with open(submit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    st = os.stat(submit_path)
    os.chmod(submit_path, st.st_mode | 0o111)

    return submit_name

def stage_missing(workspace : str, scheduler : str, files_per_job : int, mem : str, exec : bool, check_j : int) -> int:
    workspace = os.path.abspath(workspace)
    # strip trailing slash if present for nicer output
    if workspace.endswith("/"):
        workspace = workspace[:-1]
   
    metadata, tables = infer_workspace_metadata(workspace)

    print("Using metadata from workspace config:")
    print("  Runtag      :", metadata["runtag"])
    print("  Dataset     :", metadata["dataset"])
    print("  Objsyst     :", metadata["objsyst"])
    print("  Location    :", metadata["location"])
    print("  ConfigSuite :", metadata["config_suite"])
    print("  Tables      :", tables)
    print("  Scheduler   :", scheduler.upper())

    missing_files, _ = check_workspace(
            workspace,
            False,
            check_j,
        )
    
    if len(missing_files) == 0:
        print("No missing files found. Nothing to stage.")
        return 0
    
    suffix = choose_next_suffix(workspace, scheduler)
        
    # Create directories for scheduler-specific outputs
    if scheduler == "slurm":
        os.makedirs(os.path.join(workspace, "slurm"), exist_ok=True)
    elif scheduler == "condor":
        os.makedirs(os.path.join(workspace, "condor"), exist_ok=True)
    else:  # local
        os.makedirs(os.path.join(workspace, "local"), exist_ok=True)

    target_name = make_missing_target_file(workspace, suffix, list(missing_files))
    skimscript_name = make_missing_skimscript(workspace, suffix)

    job_name = f"{os.path.basename(workspace)}_missing"
    
    if scheduler == "slurm":
        submit_name = make_missing_slurm_submit(
            workspace=workspace,
            name=job_name,
            suffix=suffix,
            files_per_job=files_per_job,
            nfiles=len(missing_files),
            skimscript_name=skimscript_name,
            mem=mem,
        )
        if exec:
            cmd = ["sbatch", submit_name]
            output = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
            if output.stdout:
                print(output.stdout)
            if output.stderr:
                print(output.stderr)
            if output.returncode != 0:
                raise RuntimeError("Failed to submit missing-file SLURM jobs")
        else:
            print("Submit with:")
            print("  sbatch %s" % os.path.join(workspace, submit_name))

    elif scheduler == "condor":
        submit_name, exec_name = make_missing_condor_submit(
            workspace=workspace,
            name=job_name,
            suffix=suffix,
            files_per_job=files_per_job,
            nfiles=len(missing_files),
            skimscript_name=skimscript_name,
            mem=mem,
        )

        if exec:
            cmd = ["condor_submit", submit_name]
            output = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
            if output.stdout:
                print(output.stdout)
            if output.stderr:
                print(output.stderr)
            if output.returncode != 0:
                raise RuntimeError("Failed to submit missing-file Condor jobs")
        else:
            print("Submit with:")
            print("  condor_submit %s" % os.path.join(workspace, submit_name))

    else:  # local
        submit_name = make_missing_local_submit(
            workspace=workspace,
            suffix=suffix,
            nfiles=len(missing_files),
            skimscript_name=skimscript_name,
        )

        if exec:
            print("Running local missing-file script...")
            cmd = ["bash", submit_name]
            output = subprocess.run(cmd, cwd=workspace)
            if output.returncode != 0:
                raise RuntimeError("Failed to execute local missing-file script")
        else:
            print("Run with:")
            print("  bash %s" % os.path.join(workspace, submit_name))

    return len(missing_files)

def stage_all_missing(workspace_dir : str, 
                      scheduler: str, files_per_job : int, 
                      mem : str, exec : bool, check_j : int,
                      filter : list[str], anti_filter : list[str], only_new : bool) -> tuple[list[str], list[tuple[str, int]]]:
    
    subdirs = os.listdir(workspace_dir)
    workspaces = []
    for sd in subdirs:
        fullpath = os.path.abspath(os.path.join(workspace_dir, sd))
        if filter and any(substr not in sd for substr in filter):
            continue
        if anti_filter and any(substr in sd for substr in anti_filter):
            continue
        if is_workspace(fullpath):
            if only_new and has_existing_submission_dir(fullpath):
                print("Skipping already-submitted workspace %s" % fullpath)
                continue
            workspaces.append(fullpath)

    if len(workspaces) == 0:
        print("No skimming workspaces found in %s" % workspace_dir)
        return [], []

    # randomize the order of the workspaces to avoid overloading the scheduler with similar jobs at the same time if the input directory is sorted in some way (e.g. by dataset or date)
    import random
    random.shuffle(workspaces)

    failures = []
    nmissing = []
    for ws in workspaces:
        try:
            print("=============================================")
            print("Staging missing files for workspace %s" % ws)
            print("=============================================")

            missing = stage_missing(
                workspace=ws,
                scheduler=scheduler,
                files_per_job=files_per_job,
                mem=mem,
                exec=exec,
                check_j=check_j,
            )
            if missing > 0:
                nmissing.append((ws, missing))
        except Exception as e:
            print(f"[ERROR] Failed to stage missing files for workspace {ws}: {e}")
            failures.append(ws)

    if len(failures) > 0:
        print("Failed to stage missing files for the following %d workspaces:" % len(failures))
        for f in failures:
            print("  ", f)
    else:
        print("Successfully staged missing files for all %d workspaces." % len(workspaces))
    
    return failures, nmissing