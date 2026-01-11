import hist
from tqdm import tqdm

def build_transfer_hist(gencfg, recocfg):
    axes = []
    for axis_cfg in gencfg:
        if axis_cfg['type'] == 'Regular':
            axes.append(hist.axis.Regular(
                axis_cfg['bins'],
                axis_cfg['start'],
                axis_cfg['stop'],
                name=axis_cfg['name']+"_gen",
                transform=axis_cfg.get('transform', None),
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        elif axis_cfg['type'] == 'Variable':
            axes.append(hist.axis.Variable(
                axis_cfg['edges'],
                name=axis_cfg['name']+"_gen",
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        else:
            raise ValueError(f"Unknown axis type: {axis_cfg['type']}")
    for axis_cfg in recocfg:
        if axis_cfg['type'] == 'Regular':
            axes.append(hist.axis.Regular(
                axis_cfg['bins'],
                axis_cfg['start'],
                axis_cfg['stop'],
                name=axis_cfg['name']+"_reco",
                transform=axis_cfg.get('transform', None),
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        elif axis_cfg['type'] == 'Variable':
            axes.append(hist.axis.Variable(
                axis_cfg['edges'],
                name=axis_cfg['name']+"_reco",
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        else:
            raise ValueError(f"Unknown axis type: {axis_cfg['type']}") 
    H = hist.Hist(*axes, storage=hist.storage.Weight())
    names = list(H.axes.name)
    return H, names

def build_hist(cfg):
    axes = []
    for axis_cfg in cfg:
        if axis_cfg['type'] == 'Regular':
            axes.append(hist.axis.Regular(
                axis_cfg['bins'],
                axis_cfg['start'],
                axis_cfg['stop'],
                name=axis_cfg['name'],
                transform=axis_cfg.get('transform', None),
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        elif axis_cfg['type'] == 'Variable':
            axes.append(hist.axis.Variable(
                axis_cfg['edges'],
                name=axis_cfg['name'],
                underflow=axis_cfg.get('underflow', True),
                overflow=axis_cfg.get('overflow', True)
            ))
        else:
            raise ValueError(f"Unknown axis type: {axis_cfg['type']}")
        
    H = hist.Hist(*axes, storage=hist.storage.Weight())
    names = list(H.axes.name)
    return H, names

def get(batch, name):
    return batch[name].to_numpy(zero_copy_only=False)

def fill_hist(H, names, ds, weightname):
    iterator = tqdm(ds.to_batches(
        columns = names + [weightname, 'event_id'],
        batch_readahead = 2,
        fragment_readahead = 2,
        use_threads = True,
        batch_size = 1<<20
    ))

    total_rows = ds.count_rows()
    rows_so_far = 0

    for batch in iterator:
        rows_so_far += batch.num_rows
        iterator.set_description("Rows: %g%%" % (rows_so_far / total_rows * 100))

        H.fill(
            **{name: get(batch, name) for name in names},
            weight = get(batch, weightname)
        )

    return H
