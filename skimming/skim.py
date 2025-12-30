from skimming.objects.AllObjects import AllObjects
from skimming.selections.factories import runEventSelection, runJetSelection
from skimming.weights.factory import runWeightsFactory
from skimming.tables.driver import TableDriver
import awkward as ak

def skim(events : ak.Array, config : dict, output_path : str, tables):
    if 'count' in tables and len(tables) != 1:
        raise RuntimeError("When 'count' table is requested, it must be the only table (uses different short-circuit logic).")
    elif 'count' in tables:
        import os
        import json

        # Special short-circuit logic for count table - just count the number of events!
        n_events = len(events)
        uniqueid = AllObjects.get_uniqueid(events)
        destination = os.path.join(
            output_path,
            "count",
            uniqueid + ".json"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, 'w') as f:
            json.dump({"n_events": n_events}, f, indent=4)
        print(f"Wrote count table with {n_events} events to {destination}")
    else:
        objs = AllObjects(
            events,
            "MC",
            config['objects'], 
            config['btagging'],
            config['JERC'],
            objsyst="nominal"
        )
        eventselection = runEventSelection(
            config['eventsel'],
            objs,
            flags={}
        )
        jetselection = runJetSelection(
            config['jetsel'],   
            objs,
            eventselection, 
            flags={}
        )
        weights = runWeightsFactory(
            config['eventweight'], 
            config['eventsel'], 
            objs
        )

        driver = TableDriver(
            [
                'AK4JetKinematicsTable',
                'ConstituentKinematicsTable',
                'CutflowTable',
                'EventKinematicsTable',
                'SimonJetKinematicsTable'
            ],
            'test_output'
        )
        driver.run_tables(
            objs,
            eventselection,
            jetselection,
            weights
        )