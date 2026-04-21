import os
import shlex
import shutil
from typing import Optional

from general.fslookup.hist_lookup import get_hist_path


def _build_command(
    runtag: str,
    dataset: str,
    objsyst: str,
    wtsyst: str,
    table: str,
    location: str,
    config_suite: str,
    statN: int,
    statK: int,
    bincfg: Optional[str],
    cov: bool,
) -> str:
    parts = [
        "bin.py",
        shlex.quote(runtag),
        shlex.quote(dataset),
        shlex.quote(objsyst),
        shlex.quote(wtsyst),
        shlex.quote(table),
        "--location",
        shlex.quote(location),
        "--config-suite",
        shlex.quote(config_suite),
        "--statN",
        str(statN),
        "--statK",
        str(statK),
        "--nocheck",
    ]

    if bincfg is not None:
        parts.extend(["--bincfg", shlex.quote(bincfg)])

    if cov:
        parts.append("--cov")

    return " ".join(parts)


def setup_binning_workspace(
    working_dir: str,
    runtag: str,
    dataset_objsyst_wtsyst_triples: list[tuple[str, str, str]],
    tables: list[str],
    location: str,
    config_suite: str,
    statN: int,
    statK: int,
    bincfg: Optional[str] = None,
    cov: bool = False,
    nocheck: bool = False,
) -> int:
    this_dir = os.path.dirname(__file__)

    commands: list[str] = []
    skipped_existing = 0

    for dataset, objsyst, wtsyst in dataset_objsyst_wtsyst_triples:
        for table in tables:
            if not nocheck:
                fs, outpath = get_hist_path(
                    location,
                    config_suite,
                    runtag,
                    dataset,
                    objsyst,
                    wtsyst,
                    table,
                    cov,
                    statN,
                    statK,
                )
                if fs.exists(outpath):
                    skipped_existing += 1
                    continue

            cmd = _build_command(
                runtag=runtag,
                dataset=dataset,
                objsyst=objsyst,
                wtsyst=wtsyst,
                table=table,
                location=location,
                config_suite=config_suite,
                statN=statN,
                statK=statK,
                bincfg=bincfg,
                cov=cov,
            )
            commands.append(cmd)

    os.makedirs(working_dir, exist_ok=True)

    mode = "w"
    with open(os.path.join(working_dir, "commands.txt"), mode) as f:
        for cmd in commands:
            f.write(cmd)
            f.write("\n")

    shutil.copyfile(
        os.path.join(this_dir, "binscript.py"),
        os.path.join(working_dir, "binscript.py"),
    )

    return len(commands)
