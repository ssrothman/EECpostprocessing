from __future__ import annotations
from binning.main import build_hist, build_transfer_config, fill_cov, fill_hist
from general.datasets.datasets import location_lookup
from general.fslookup.location_lookup import lookup_hostid
from general.fslookup.skim_path import lookup_skim_path
from general.fslookup.hist_lookup import get_hist_path, get_hist_bincfg_path
from typing import Any
import os.path
import json
import uuid
from simonpy.AbitraryBinning import ArbitraryBinning, ArbitraryGenRecoBinning
import pyarrow.dataset as ds
import numpy as np

def run_table(args : Any, table: str):
    bincfg_name = args.bincfg
    if bincfg_name is None:
        bincfg_name = '_'.join(table.split('_')[:-1])

    # load binning config
    binpkgpath = os.path.dirname(__file__)
    bincfgpath = os.path.join(
        binpkgpath,
        'config',
        bincfg_name
    )
    with open(bincfgpath + '.json') as f:
        bincfg = json.load(f)

    if table.endswith('Reco'):
        thebinning = bincfg['reco']
    elif table.endswith('Gen'):
        thebinning = bincfg['gen']
    elif table.endswith('transfer'):
        thebinning = build_transfer_config(
            bincfg['gen'],
            bincfg['reco']
        )
    else:
        raise NotImplementedError("Only gen, reco, transfer can be pre-binned!")

    fs, skimpath = lookup_skim_path(
        args.location,
        args.config_suite,
        args.runtag,
        args.dataset,
        args.objsyst,
        table
    )

    H, prebinned = build_hist(thebinning)
    print("Reading dataset from", skimpath)
    dataset = ds.dataset(skimpath, format='parquet', filesystem=fs)

    if table.endswith('transfer'):
        itemwt = 'wt_reco'
        ab = ArbitraryGenRecoBinning()
        Hreco, prebinned_reco = build_hist(bincfg['reco'])
        Hgen, prebinned_gen = build_hist(bincfg['gen'])
        ab.setup_from_histograms(
            Hreco,
            Hgen
        )
    else:
        itemwt = 'wt'
        ab = ArbitraryBinning()
        ab.setup_from_histogram(H)

    if args.cov:
        thefun = fill_cov
    else:
        thefun = fill_hist

    result = thefun(
        H, prebinned,
        dataset,
        'wt_%s' % args.wtsyst,
        itemwt = itemwt,
        statN = args.statN,
        statK = args.statK,
        reweight = None #for now
    )

    if args.cov:
        halfshape = result.shape[:len(result.shape)//2]
        thelen = np.prod(halfshape)
        output = result.reshape((thelen, thelen)) # type: ignore
    else:
        output = result.values(flow=True).ravel() # type: ignore
        
    _, outpath = get_hist_path(
        args.location,
        args.config_suite,
        args.runtag,
        args.dataset,
        args.objsyst,
        args.wtsyst,
        table,
        args.cov,
        args.statN,
        args.statK
    )

    print("Writing result to", outpath)
    tmp_outpath = f"{outpath}.tmp.{uuid.uuid4().hex}"
    try:
        with fs.open(tmp_outpath, 'wb') as f:
            np.save(f, output)

        # Validate by reading back from the temporary file.
        with fs.open(tmp_outpath, 'rb') as f:
            loaded_output = np.load(f, allow_pickle=False)
        if loaded_output.shape != output.shape or loaded_output.dtype != output.dtype:
            raise RuntimeError("Validation failed for temporary histogram output at %s" % tmp_outpath)

        # Copy to final location (xrootd-compatible alternative to mv)
        with fs.open(tmp_outpath, 'rb') as src:
            with fs.open(outpath, 'wb') as dst:
                dst.write(src.read())
        
        # validate again after copying to final location
        import fsspec.implementations.local
        from simonpy.checksum import checksum_file

        tmpcheck = checksum_file(tmp_outpath, fs)
        finalcheck = checksum_file(outpath, fs)

        if tmpcheck != finalcheck:
            print("tmp checksum:", tmpcheck)
            print("final checksum:", finalcheck)
            raise RuntimeError("Validation failed for histogram output after copying to final location at %s" % outpath)
        
        fs.rm(tmp_outpath)
    except Exception:
        if fs.exists(tmp_outpath):
            fs.rm(tmp_outpath)
        raise

    _, bincfg_path = get_hist_bincfg_path(
        args.location,
        args.config_suite,
        args.runtag,
        args.dataset,
        args.objsyst,
        table
    )

    if fs.exists(bincfg_path):
        oldbinning = type(ab)() # use same type as ab [either ArbitraryBinning or ArbitraryGenRecoBinning]

        with fs.open(bincfg_path, 'r') as f:
            oldbinning.from_dict(json.load(f))
        
        if oldbinning != ab:
            raise RuntimeError("Binning config mismatch for existing bincfg at %s" % bincfg_path)
    else:
        print("Writing binning config to", bincfg_path)
        tmp_bincfg_path = f"{bincfg_path}.tmp.{uuid.uuid4().hex}"
        bincfg_dict = ab.to_dict()
        try:
            with fs.open(tmp_bincfg_path, 'w') as f:
                json.dump(bincfg_dict, f, indent=4)

            # Validate by reading back from the temporary file.
            with fs.open(tmp_bincfg_path, 'r') as f:
                loaded_bincfg = json.load(f)
            if loaded_bincfg != bincfg_dict:
                raise RuntimeError("Validation failed for temporary bincfg output at %s" % tmp_bincfg_path)

            # Copy to final location (xrootd-compatible alternative to mv)
            with fs.open(tmp_bincfg_path, 'r') as src:
                with fs.open(bincfg_path, 'w') as dst:
                    dst.write(src.read()) # type: ignore

            # validate again after copying
            with fs.open(bincfg_path, 'r') as f:
                loaded_bincfg = json.load(f)
            if loaded_bincfg != bincfg_dict:
                raise RuntimeError("Validation failed for bincfg after copying to final location at %s" % bincfg_path)
            
            fs.rm(tmp_bincfg_path)

        except Exception:
            if fs.exists(tmp_bincfg_path):
                fs.rm(tmp_bincfg_path)
            raise
