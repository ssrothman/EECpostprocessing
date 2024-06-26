if __name__ == '__main__':
    ################### ARGUMENT PARSING ###################
    import argparse

    parser = argparse.ArgumentParser(description='Produce histograms off of NanoAOD files')

    parser.add_argument('--statsplit', action='store_true')

    parser.add_argument("sample", type=str)
    parser.add_argument("what", type=str)
    parser.add_argument('jettype', type=str)
    parser.add_argument('EECtype', type=str)

    parser.add_argument('--treatAsData', action='store_true')

    parser.add_argument('--extra-tags', type=str, default=None, required=False, nargs='*')

    parser.add_argument('--bTag', type=str, default='tight', required=False, choices=['tight', 'medium', 'loose'])

    parser.add_argument('--noRoccoR', action='store_true')
    parser.add_argument('--noJER', action='store_true')
    parser.add_argument('--noJEC', action='store_true')
    parser.add_argument('--noPUweight', action='store_true')
    parser.add_argument('--noPrefireSF', action='store_true')
    parser.add_argument('--noIDsfs', action='store_true')
    parser.add_argument('--noIsosfs', action='store_true')
    parser.add_argument('--noTriggersfs', action='store_true')
    parser.add_argument('--noBtagSF', action='store_true')

    parser.add_argument('--Zreweight', action='store_true')

    syst_group = parser.add_mutually_exclusive_group(required=False)
    syst_group.add_argument('--nom', dest='syst', action='store_const',
                            const='nom')

    syst_group.add_argument('--JER', dest='syst', action='store_const', 
                            const='JER')
    syst_group.add_argument('--JES', dest='syst', action='store_const',
                            const='JES')

    syst_group.add_argument("--wt_prefire", dest='syst', action='store_const',
                            const='wt_prefire')
    syst_group.add_argument("--wt_idsf", dest='syst', action='store_const',
                            const='wt_idsf')
    syst_group.add_argument("--wt_isosf", dest='syst', action='store_const',
                            const='wt_isosf')
    syst_group.add_argument("--wt_triggersf", dest='syst', action='store_const',
                            const='wt_triggersf')
    syst_group.add_argument("--wt_scale", dest='syst', action='store_const',
                            const='wt_scale')
    syst_group.add_argument("--wt_ISR", dest='syst', action='store_const',
                            const='wt_ISR')
    syst_group.add_argument("--wt_FSR", dest='syst', action='store_const',
                            const='wt_FSR')
    syst_group.add_argument("--wt_PDF", dest='syst', action='store_const',
                            const='wt_PDF')
    syst_group.add_argument("--wt_aS", dest='syst', action='store_const',
                            const='wt_aS')
    syst_group.add_argument("--wt_PDFaS", dest='syst', action='store_const',
                            const='wt_PDFaS')
    syst_group.add_argument('--wt_PU', dest='syst', action='store_const',
                            const='wt_PU')
    syst_group.add_argument('--wt_btagSF', dest='syst', action='store_const',
                            const='wt_btagSF')
    parser.set_defaults(syst='nom')

    syst_updn_group = parser.add_mutually_exclusive_group(required=False)
    syst_updn_group.add_argument('--DN', dest='syst_updn', 
                                 action='store_const', const='DN')
    syst_updn_group.add_argument('--UP', dest='syst_updn',
                                 action='store_const', const='UP')
    parser.set_defaults(syst_updn=None)

    parser.add_argument("--local", action='store_true')

    parser.add_argument('--nfiles', dest='nfiles', type=int, help='number of files to process', default=None, required=False)
    parser.add_argument('--startfile', type=int, default=0, required=False)

    scale_group = parser.add_mutually_exclusive_group(required=False)
    scale_group.add_argument('--use-slurm', dest='scale', action='store_const', const='slurm')
    scale_group.add_argument('--use-local', dest='scale', action='store_const', const='local')
    scale_group.add_argument('--use-local-debug', dest='scale', action='store_const', const='local_debug')
    parser.set_defaults(scale=None)
    args = parser.parse_args()

    if args.syst != 'nom' and args.syst_updn is None:
        raise ValueError("Must specify UP or DN for systematic")

    ######################################################################

    from processing.EECProcessor import EECProcessor
    from processing.scaleout import setup_cluster_on_submit, custom_scale, setup_htcondor

    from reading.files import get_rootfiles

    from RecursiveNamespace import RecursiveNamespace

    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
    from coffea.processor import Runner, IterativeExecutor, FuturesExecutor, DaskExecutor

    import os
    import json

    from samples.latest import SAMPLE_LIST

    ################### INPUT ###################

    if args.local:
        files = [args.sample]
    else:
        sample = SAMPLE_LIST.lookup(args.sample)
        files = sample.get_files()
        if args.nfiles is not None:
            files = files[args.startfile:args.nfiles+args.startfile]

    print("Processing %d files"%len(files))
    print(files[0])

    ##############################################

    ################### PROCESSOR ###################
    with open("configs/base.json", 'r') as f:
        config = RecursiveNamespace(**json.load(f))

    with open("configs/%s.json"%args.jettype, 'r') as f:
        config.update(json.load(f))

    with open("configs/%sEEC.json"%args.EECtype, 'r') as f:
        config.update(json.load(f))

    config.tagging.wp = args.bTag

    argsdict = {
        'config' : config,
        'statsplit' : args.statsplit,
        'what' : args.what,
        'syst' : args.syst,
        'syst_updn' : args.syst_updn,
        'era' : '2018A' if args.local else sample.JEC,
        'flags' : None if args.local else sample.flags,
        'noRoccoR' : args.noRoccoR,
        'noJER' : args.noJER,
        'noJEC' : args.noJEC,
        'noPUweight' : args.noPUweight,
        'noPrefireSF' : args.noPrefireSF,
        'noIDsfs' : args.noIDsfs,
        'noIsosfs' : args.noIsosfs,
        'noTriggersfs' : args.noTriggersfs,
        'noBtagSF' : args.noBtagSF,
        'Zreweight' : args.Zreweight,
        'treatAsData' : args.treatAsData
    }

    def process_func(file, args):
        events = NanoEventsFactory.from_root(file).events()
        processor_instance = EECProcessor(**args)
        return processor_instance.process(events)

    ##################################################

    ################### OUTPUT ###################
    out_fname = 'hists'
    if args.syst != 'nom':
        if args.syst.startswith('wt_'):
            out_fname += '_%s%s'%(args.syst[3:], args.syst_updn)
        else:
            out_fname += '_%s%s'%(args.syst, args.syst_updn)

    out_fname += '_file%dto%d'%(args.startfile, args.startfile+len(files))

    if args.bTag == 'tight':
        out_fname += '_tight'
    elif args.bTag == 'medium':
        out_fname += '_medium'
    elif args.bTag == 'loose':
        out_fname += '_loose'

    if args.statsplit:
        out_fname += '_statsplit'
    if args.noRoccoR:
        out_fname += '_noRoccoR'
    if args.noJER:
        out_fname += '_noJER'
    if args.noJEC:
        out_fname += '_noJEC'
    if args.noPUweight:
        out_fname += '_noPUweight'
    if args.noPrefireSF:
        out_fname += '_noPrefireSF'
    if args.noIDsfs:
        out_fname += '_noIDsfs'
    if args.noIsosfs:
        out_fname += '_noIsosfs'
    if args.noTriggersfs:
        out_fname += '_noTriggersfs'
    if args.noBtagSF:
        out_fname += '_noBtagSF'
    if args.Zreweight:
        out_fname += '_Zreweight'

    if args.extra_tags is not None:
        for tag in args.extra_tags:
            out_fname += '_%s'%tag

    if args.treatAsData:
        out_fname += '_asData'

    out_fname += '.pkl'

    if args.local:
        destination = 'testlocal'
    else:
        destination = "/data/submit/srothman/EEC/%s/%s/%s"%(SAMPLE_LIST.tag, sample.name, args.what)
        if os.path.exists(os.path.join(destination, out_fname)):
            raise ValueError("Destination %s already exists"%os.path.join(destination, out_fname))

    print("Outputting to %s"%os.path.join(destination, out_fname))

    os.makedirs(destination, exist_ok=True)
    ##################################################

    ################### EXECUTION ###################

    if args.scale is None:
        if len(files) == 1:
            args.scale = 'local_debug'
        elif len(files) < 10:
            args.scale = 'local'
        else:
            args.scale = 'slurm'

    if args.scale == 'slurm':
        cluster, client = setup_cluster_on_submit(1, 200, destination)
    elif args.scale == 'local':
        from dask.distributed import LocalCluster, Client
        cluster = LocalCluster(n_workers=10, 
                               threads_per_worker=1,
                               dashboard_address=9876)
        client = cluster.get_client()
    elif args.scale == 'local_debug':
        from dask.distributed import LocalCluster, Client
        cluster = LocalCluster(n_workers=1, 
                               threads_per_worker=1,
                               dashboard_address=9876)
        client = cluster.get_client()

    print(client.dashboard_link)
    ##################################################

    ################### RUNNING ###################

    futures = client.map(process_func, files, args=argsdict)

    from dask.distributed import as_completed
    from tqdm import tqdm
    from coffea.processor.accumulator import iadd

    final_ans = None
    t = tqdm(as_completed(futures), total=len(futures),
             desc='Events: 0')
    for ans in t:
        try:
            if final_ans is None:
                final_ans = ans.result()
            else:
                iadd(final_ans, ans.result())
            t.set_description("Events: %g"%final_ans['sumwt'], refresh=True)
        except:
            print("Error in processing")
            print(ans.result())
            continue
        finally:
            ans.release()

    with open(os.path.join(destination,out_fname), 'wb') as fout:
        import pickle
        pickle.dump(final_ans, fout)

    client.close()
    cluster.close()

    ##################################################
