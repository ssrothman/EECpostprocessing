if __name__ == '__main__':
    from coffea.nanoevents import NanoEventsFactory
    import numpy as np
    import awkward as ak
    from reading.files import get_rootfiles
    from RecursiveNamespace import RecursiveNamespace
    from reading.allreader import AllReaders
    from tqdm import tqdm
    import json
    from processing.scaleout import setup_cluster_on_submit

    datafiles = get_rootfiles('cmseos.fnal.gov','/store/group/lpcpfnano/srothman/Jun11_2024_res4_notransfer/2018/SingleMuon')
    pythiafiles = get_rootfiles('cmseos.fnal.gov', '/store/group/lpcpfnano/srothman/Jun11_2024_res4_notransfer/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8')
    herwigfiles = get_rootfiles('cmseos.fnal.gov', '/store/group/lpcpfnano/srothman/Jun11_2024_res4_notransfer/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7')

    with open("configs/base.json", 'r') as f:
        config = RecursiveNamespace(**json.load(f))
    with open("configs/ak8.json", 'r') as f:
        config.update(json.load(f))
    with open("configs/inclusiveEEC.json", 'r') as f:
        config.update(json.load(f))

    def doOneData(file):
        return doOne(file, onGen=False)

    def doOne(file, onGen=True):
        evts = NanoEventsFactory.from_root(file).events()

        readers = AllReaders(evts, config,
                             noRoccoR = False,
                             noJER = True, noJEC = True)

        if onGen:
            ptmask = readers.rGenJet.jets.pt > 500
            ptmask = ptmask[readers.rGenEEC.iJet]
            EEC = readers.rGenEEC.res4tee
        else:
            ptmask = readers.rRecoJet.jets.pt > 500
            ptmask = ptmask[readers.rRecoEEC.iJet]
            EEC = readers.rRecoEEC.res4tee

        if ak.sum(ptmask) == 0:
            return np.zeros((10,21,21))
        #should have shape (nshape, RL, r, phi)
        return ak.to_numpy(ak.sum(ak.sum(EEC[ptmask], axis=0), axis=0))

    #import dask
    #lazy_fn = dask.delayed(doOne)
    #lazy_ans = [lazy_fn(file) for file in herwigfiles[:10]]

    import dask
    import dask.distributed
    #client = dask.distributed.Client(threads_per_worker=1, n_workers=25)
    cluster, client = setup_cluster_on_submit(100, 200)

    '''print("DOING HERWIG")
    futures = client.map(doOne, herwigfiles)

    actual_ans = np.zeros((10,21,21))
    for ans in tqdm(dask.distributed.as_completed(futures), total=len(herwigfiles)):
        if np.sum(ans.result()) == 0:
            continue
        actual_ans += ans.result()

    np.save("herwig_res4.npy", actual_ans)'''

    print("DOING PYTHIA")
    futures = client.map(doOne, pythiafiles)

    actual_ans = np.zeros((10,21,21))
    for ans in tqdm(dask.distributed.as_completed(futures), total=len(pythiafiles)):
        try:
            if np.sum(ans.result()) == 0:
                continue
            actual_ans += ans.result()
        except:
            print("ONE JOB FAILED")
            continue

    np.save("pythia_res4.npy", actual_ans)

    print("DOING DATA")
    futures = client.map(doOneData, datafiles)

    actual_ans = np.zeros((10,21,21))
    for ans in tqdm(dask.distributed.as_completed(futures), total=len(datafiles)):
        if np.sum(ans.result()) == 0:
            continue
        actual_ans += ans.result()

    np.save("data_res4.npy", actual_ans)
