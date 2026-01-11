from .AK4jetkinematics import AK4JetKinematicsTable
from .eventkinematics import EventKinematicsTable   
from .constituentkinematics import ConstituentKinematicsTable
from .cutflow import CutflowTable
from .jetkinematics import SimonJetKinematicsTable
from .EEC import EECres4ObsTable, EECres4TransferTable

from coffea.analysis_tools import Weights, PackedSelection
from skimming.objects.AllObjects import AllObjects
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any

import os.path

table_classes = {
    "AK4JetKinematicsTable": AK4JetKinematicsTable,
    "EventKinematicsTable": EventKinematicsTable,
    "ConstituentKinematicsTable": ConstituentKinematicsTable,
    "CutflowTable": CutflowTable,
    "SimonJetKinematicsTable": SimonJetKinematicsTable,
    "EECres4Obs": EECres4ObsTable,
    "EECres4Transfer": EECres4TransferTable,
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
        print("parsing option:", opt)
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
        print("  interpreted as:", opt)
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
        if isinstance(result, dict):
            import json
            with self._fs.open(destination + ".json", "w") as f:
                json.dump(result, f, indent=4)
        else:
            import pyarrow.parquet as pq
            pq.write_table(result, destination + ".parquet", filesystem=self._fs)

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
