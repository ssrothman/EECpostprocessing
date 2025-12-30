
from skimming.datasets.datasets import get_JERC_era, get_flags
from skimming.fsutil.location_lookup import location_lookup


def setup_skim_workspace(working_dir, 
                        runtag, dataset, 
                        objsyst,
                        config, tables,
                        output_location):
    from skimming.datasets.datasets import get_target_files
    import os.path
    import json
    import os

    os.makedirs(working_dir, exist_ok=True)

    if len(tables) == 1 and tables[0] == 'count':
        exclude_dropped = False
    else:
        exclude_dropped = True

    target_files, location = get_target_files(runtag, dataset, exclude_dropped=exclude_dropped)

    with open(os.path.join(working_dir, 'target_files.txt'), 'w') as f:
        for tf in target_files:
            f.write(f"{tf}\n")

    config['output_location'] = output_location
    config['output_path'] = os.path.join(
        config['configsuite_name'], runtag, dataset, objsyst
    )
    config['objsyst'] = objsyst
    config['tables'] = tables
    config['input_location'] = location

    config['era'] = get_JERC_era(runtag, dataset)
    config['flags'] = get_flags(runtag, dataset)

    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    #also write config into output destination for record-keeping purposes
    output_fs, output_basepath = location_lookup(output_location)
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