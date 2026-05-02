from .AK4jetkinematics import AK4JetKinematicsTable
from .eventkinematics import EventKinematicsTable   
from .constituentkinematics import ConstituentKinematicsTable
from .cutflow import CutflowTable
from .jetkinematics import SimonJetKinematicsTable
from .EEC import EECres4ObsTable, EECres4TransferTable
from .generictable import GenericTable

from coffea.analysis_tools import Weights, PackedSelection
from skimming.objects.AllObjects import AllObjects
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any

import os.path
import hashlib
import tempfile

table_classes = {
    "AK4JetKinematicsTable": AK4JetKinematicsTable,
    "EventKinematicsTable": EventKinematicsTable,
    "ConstituentKinematicsTable": ConstituentKinematicsTable,
    "CutflowTable": CutflowTable,
    "SimonJetKinematicsTable": SimonJetKinematicsTable,
    "EECres4Obs": EECres4ObsTable,
    "EECres4Transfer": EECres4TransferTable,
    'GenericTable': GenericTable
}

def construct_table_from_string(table_str : str) -> Any:
    if ':' in table_str:
        tablename, options_str = table_str.split(':')
        options : list[Any] = options_str.split(',')
    else:
        tablename = table_str
        options : list[Any]= []

    #coerce datatypes
    for i in range(len(options)):
        opt = options[i].strip()
        if opt.lower() == 'true':
            opt = True
        elif opt.lower() == 'false':
            opt = False
        else:
            try:
                opt = int(opt)
            except ValueError:
                try:
                    opt = float(opt)
                except ValueError:
                    pass
        options[i] = opt

    if tablename not in table_classes:
        raise ValueError(f"Unknown table class '{tablename}'")
    
    table_class = table_classes[tablename]
    return table_class(*options)

class TableDriver:
    def __init__(self, 
                 tables : list[str], 
                 basepath : str,
                 fs : Any):
        self._tables = [construct_table_from_string(t) for t in tables]
        self._basepath = basepath
        self._fs = fs
        
    def _run_one_table(self, 
                       table_obj : Any,
                       objs : AllObjects, 
                       evtsel : PackedSelection, 
                       jetsel : PackedJetSelection, 
                       weights : Weights):

        print("Running table:", table_obj.name)

        destination = os.path.join(
            self._basepath,
            table_obj.name,
            objs.uniqueid
        )
        self._fs.makedirs(os.path.dirname(destination), exist_ok=True)
        print("\tOutput path:", destination)

        result = table_obj.run_table(
            objs, evtsel, jetsel, weights
        )
        print("Ran table:", table_obj.name)
        if isinstance(result, dict):
            import json
            print("Dumping result as JSON")
            # Atomic write: temp file + copy + validation
            temp_dest = destination + ".tmp"
            final_dest = destination + ".json"
            try:
                with self._fs.open(temp_dest, "w") as f:
                    json.dump(result, f, indent=4)
                # Verify by reading back
                with self._fs.open(temp_dest, "r") as f:
                    _ = json.load(f)
                print("JSON validation passed, moving to final destination")
                
                # mv is not supported on all filesystems, so we do a copy + delete
                with self._fs.open(temp_dest, "r") as src:
                    with self._fs.open(final_dest, "w") as dst:
                        dst.write(src.read())
                        if hasattr(dst, 'flush'):
                            dst.flush()
                
                # validate after copying to final destination
                # use checksums
                tmpcheck = self._fs.checksum(temp_dest)
                finalcheck = self._fs.checksum(final_dest)
                if tmpcheck != finalcheck:
                    raise ValueError("Checksum mismatch after copying JSON to final destination")
                
                # cleanup temp file
                self._fs.rm(temp_dest)

            except Exception as e:
                print(f"ERROR writing/validating JSON for {table_obj.name}: {e}")
                try:
                    self._fs.rm(temp_dest)
                except:
                    pass
                raise
        else:
            print(len(result), "rows")
            if len(result) == 0:
                print("WARNING: result is empty for table", table_obj.name)
                result = result.select([])  # create empty table with correct schema
            import pyarrow.parquet as pq
            print("Dumping result as Parquet")
            
            # Atomic write: temp file + rename + validation
            temp_dest = destination + ".tmp"
            final_dest = destination + ".parquet"
            try:
                with self._fs.open(temp_dest, "wb") as f:
                    print("Writing Parquet file to temporary location:", temp_dest)
                    pq.write_table(result, f)
                    if hasattr(f, 'flush'):
                        f.flush()
                
                # Validate by reading back the metadata
                print("Validating written Parquet file...")
                with self._fs.open(temp_dest, "rb") as f:
                    pf = pq.ParquetFile(f)
                    actual_rows = pf.metadata.num_rows
                    if actual_rows != len(result):
                        raise ValueError(
                            f"Row count mismatch: wrote {len(result)} rows, "
                            f"but read back {actual_rows} rows"
                        )
                    # Try to read the file to ensure it's not corrupted
                    _ = pf.read()  
                
                print(f"Parquet validation passed ({actual_rows} rows), moving to final destination")

                # mv is not supported on all filesystems, so we do a copy + delete
                with self._fs.open(temp_dest, "rb") as src:
                    with self._fs.open(final_dest, "wb") as dst:
                        dst.write(src.read())
                        if hasattr(dst, 'flush'):
                            dst.flush()

                # validate again after copying to final destination
                # use checksums
                from simonpy.checksum import checksum_file
                tmpcheck = checksum_file(temp_dest, self._fs)
                finalcheck = checksum_file(final_dest, self._fs)

                if tmpcheck != finalcheck:
                    print("tmp checksum:", tmpcheck)
                    print("final checksum:", finalcheck)
                    raise ValueError("Checksum mismatch after copying to final destination")
                
                # cleanup temp file
                self._fs.rm(temp_dest)

            except Exception as e:
                print(f"ERROR writing/validating Parquet for {table_obj.name}: {e}")
                try:
                    self._fs.rm(temp_dest)
                except:
                    pass
                raise

    def run_tables(self,
                   objs,
                   evtsel,
                   jetsel,
                   weights):
        for table_obj in self._tables:
            self._run_one_table(
                table_obj,
                objs,
                evtsel,
                jetsel,
                weights
            )
