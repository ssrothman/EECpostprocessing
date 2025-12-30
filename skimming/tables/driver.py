from .AK4jetkinematics import AK4JetKinematicsTable
from .eventkinematics import EventKinematicsTable   
from .constituentkinematics import ConstituentKinematicsTable
from .cutflow import CutflowTable
from .jetkinematics import SimonJetKinematicsTable

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
}

class TableDriver:
    def __init__(self, 
                 tables : list[str], 
                 basepath : str,
                 fs : Any):
        self._tables = [table_classes[t]() for t in tables]
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
