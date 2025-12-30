
def make_skimscript(working_dir, 
                    runtag, dataset, 
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
        runtag, dataset
    )
    config['tables'] = tables
    config['input_location'] = location

    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    #copy skimscript
    import shutil
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'skimscript.py'),
                    os.path.join(working_dir, 'skimscript.py'))