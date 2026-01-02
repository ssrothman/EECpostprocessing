
from skimming.datasets.datasets import get_JERC_era, get_flags
from skimming.fsutil.location_lookup import location_lookup

from skimming.tables.driver import construct_table_from_string

def setup_skim_workspace(working_dir, 
                        runtag, dataset, 
                        objsyst,
                        config, tables,
                        output_location):
    from skimming.datasets.datasets import get_target_files
    import os.path
    import json
    import os


    if len(tables) == 1 and tables[0] == 'count':
        exclude_dropped = False
    else:
        exclude_dropped = True

    target_files, location = get_target_files(runtag, dataset, exclude_dropped=exclude_dropped)

    config['output_location'] = output_location
    config['output_path'] = os.path.join(
        config['configsuite_name'], runtag, dataset, objsyst
    )
    config['objsyst'] = objsyst
    config['tables'] = tables
    config['input_location'] = location

    output_fs, output_basepath = location_lookup(output_location)

    n_targets = len(target_files)
    if n_targets == 0:
        raise RuntimeError(f"No target files found for dataset {dataset} with runtag {runtag}")
    
    # check if all the outputs already exist
    any_need = False
    for table in tables:
        if table == 'count':
            tablename = 'count'
        else:
            tablename = construct_table_from_string(table).name

        tpath = os.path.join(
            output_basepath,
            config['output_path'],
            tablename
        )

        if output_fs.exists(tpath):

            existing_pq = output_fs.glob(os.path.join(tpath, '*.parquet'))
            existing_json = output_fs.glob(os.path.join(tpath, '*.json'))
            if len(existing_pq) != n_targets and len(existing_json) != n_targets:
                any_need = True
                break
        else:
            any_need = True
            break

    if not any_need:
        print("All outputs for [dataset %s, objsyst %s, tables %s] already exist, skipping workspace setup." % (dataset, objsyst, tables))
        return

    os.makedirs(working_dir, exist_ok=True)

    with open(os.path.join(working_dir, 'target_files.txt'), 'w') as f:
        for tf in target_files:
            f.write(f"{tf}\n")


    config['era'] = get_JERC_era(runtag, dataset)
    config['flags'] = get_flags(runtag, dataset)

    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    #also write config into output destination for record-keeping purposes
    output_cfg_path = os.path.join(
        config['output_path'], 'config.json'
    )

    if output_fs.exists(output_cfg_path):
        with output_fs.open(output_cfg_path, 'r') as f:
            existing_cfg = json.load(f)
        
        prev_to_check = existing_cfg.copy()
        del prev_to_check['tables']
        curr_to_check = config.copy()
        del curr_to_check['tables']
        if prev_to_check != curr_to_check:
            raise ValueError("Configuration at output location differs from current configuration!")

    output_fs.makedirs(os.path.dirname(os.path.join(output_basepath, output_cfg_path)), exist_ok=True)
    with output_fs.open(os.path.join(output_basepath, output_cfg_path), 'w') as f:
        json.dump(config, f, indent=4)
        
    #copy skimscript
    import shutil
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'skimscript.py'),
                    os.path.join(working_dir, 'skimscript.py'))