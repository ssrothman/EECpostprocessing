if __name__ == '__main__':
    ################### ARGUMENT PARSING ###################
    import argparse

    parser = argparse.ArgumentParser(description='Produce histograms off of NanoAOD files')

    parser.add_argument("sample", type=str)
    parser.add_argument("binningtype", type=str)
    parser.add_argument('configsuite', type=str)

    parser.add_argument('--force', action='store_true')
    parser.add_argument('--recover', action='store_true')

    parser.add_argument('--filesplit', type=int, default=1, required=False)
    parser.add_argument('--filebatch', type=int, default=1, required=False)

    parser.add_argument('--extra-tags', type=str, default=None, required=False, nargs='*')

    parser.add_argument('--samplelist', type=str, default='latest', required=False)

    parser.add_argument('--syst', type=str, default='nominal', 
                        choices=['nominal', 'JES_UP', 'JES_DN',
                                 'JER_UP', 'JER_DN',
                                 'UNCLUSTERED_UP',
                                 'UNCLUSTERED_DN'])

    parser.add_argument('--noBkgVeto', action='store_true')
    parser.add_argument('--noRoccoR', action='store_true')
    parser.add_argument('--noJER', action='store_true')
    parser.add_argument('--noJEC', action='store_true')
    parser.add_argument('--noJUNC', action='store_true')
    parser.add_argument('--noPUweight', action='store_true')
    parser.add_argument('--noPrefireSF', action='store_true')
    parser.add_argument('--noIDsfs', action='store_true')
    parser.add_argument('--noIsosfs', action='store_true')
    parser.add_argument('--noTriggersfs', action='store_true')
    parser.add_argument('--noBtagSF', action='store_true')

    parser.add_argument('--Zreweight', action='store_true')

    parser.add_argument("--local", action='store_true')

    parser.add_argument('--nfiles', dest='nfiles', type=int, help='number of files to process', default=None, required=False)
    parser.add_argument('--startfile', type=int, default=0, required=False)

    scale_group = parser.add_mutually_exclusive_group(required=False)
    scale_group.add_argument('--use-slurm', dest='scale', action='store_const', const='slurm')
    scale_group.add_argument('--use-local', dest='scale', action='store_const', const='local')
    scale_group.add_argument('--use-local-debug', dest='scale', action='store_const', const='local_debug')
    parser.set_defaults(scale=None)

    parser.add_argument('--numCollectionThreads', type=int, default=8, required=False)
    parser.add_argument('--numAddingProcesses', type=int, default=8, required=False)

    parser.add_argument('--JECera', type=str, default='', required=False)
    parser.add_argument('--verbose', type=int, default=0, required=False)

    args = parser.parse_args()

    ######################################################################

    from processing.EECProcessor import EECProcessor
    from processing.scaleout import setup_cluster_on_submit, setup_local_cluster

    from RecursiveNamespace import RecursiveNamespace

    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory, BaseSchema

    import os
    import os.path
    import json

    import samples
    if not args.local:
        SAMPLE_LIST = samples.samplelists[args.samplelist].SAMPLE_LIST

    import dask
    dask.config.set({'distributed.client.heartbeat': '120s'})
    dask.config.set({'distributed.comm.retry.count': 10})
    dask.config.set({'distributed.comm.timeouts.connect': '120s'})
    dask.config.set({'distributed.comm.timeouts.tcp': '120s'})
    dask.config.set({'distributed.deploy.lost-worker-timeout': '120s'})
    dask.config.set({'distributed.scheduler.worker-saturation': 1.0})
    dask.config.set({'distributed.scheduler.locks.lease-timeout': '120s'})
    dask.config.set({'distributed.worker.memory.target' :  False})
    dask.config.set({'distributed.worker.memory.spill' :  False})
    dask.config.set({'distributed.worker.memory.pause' :  False})
    dask.config.set({'distributed.worker.memory.kill' :  False})
    dask.config.set({'distributed.worker.memory.rebalance.measure' : 'managed'})
    dask.config.set({'distributed.scheduler.active-memory-manager.measure' : 'managed'})

    #dask.config.set({'distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_': '0'})

    ################### INPUT/OUTPUT ###################
    if args.local:
        files = [args.sample]
    else:
        sample = SAMPLE_LIST.lookup(args.sample)
        files = sample.get_files(exclude_dropped = args.binningtype != 'Count')
        if args.nfiles is not None:
            files = files[args.startfile:args.nfiles+args.startfile]

    import numpy as np

    out_fname = 'hists'
    out_fname += '_file%dto%d'%(args.startfile, args.startfile+len(files))

    if args.noRoccoR:
        out_fname += '_noRoccoR'
    if args.noJER:
        out_fname += '_noJER'
    if args.noJEC:
        out_fname += '_noJEC'
    if args.noJUNC:
        out_fname += '_noJUNC'
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
    if args.noBkgVeto:
        out_fname += '_noBkgVeto'

    if args.extra_tags is not None:
        for tag in args.extra_tags:
            out_fname += '_%s'%tag

    out_fname += '_%s'%args.configsuite

    if args.local:
        destination = 'testlocal'
    else:
        destination = "/ceph/submit/data/group/cms/store/user/srothman/EEC/%s/%s/%s"%(SAMPLE_LIST.tag, sample.name, args.binningtype)
        if os.path.exists(os.path.join(destination, out_fname)) and not (args.force or args.recover):
            raise ValueError("Destination %s already exists"%os.path.join(destination, out_fname))

    if args.verbose:
        print("Outputting to %s"%os.path.join(destination, out_fname))

    os.makedirs(destination, exist_ok=True)

    if args.recover:
        import pickle
        with open(os.path.join(destination, out_fname+"_%s.pickle"%args.syst), 'rb') as f:
            prev = pickle.load(f)
        
        files = prev['errd']

    if args.verbose:
        print("Processing %d files"%len(files))
        print(files[0])

    np.random.shuffle(files)
    ##############################################

    ################### PROCESSOR ###################
    with open("configs/suites/%s.json"%args.configsuite, 'r') as f:
        configsuite = RecursiveNamespace(**json.load(f))

    config = RecursiveNamespace()
    for configname in configsuite.configs:
        with open(configname, 'r') as f:
            config.update(json.load(f))

    with open("configs/binning/%s.json"%args.binningtype, 'r') as f:
        config.update(json.load(f))

    if args.verbose:
        print("config suite", args.configsuite)
        print("binning type", args.binningtype)
        print()
        configstr = json.dumps(config.to_dict(), indent=4)
        print(configstr)
        print()

    if args.JECera == '':
        if args.local:
            JECera = 'skip'
        else:
            JECera = sample.JEC
    else:
        JECera = args.JECera

    argsdict = {
        'binningtype' : args.binningtype,
        'era' : JECera,
        'flags' : None if args.local else sample.flags,
        'noRoccoR' : args.noRoccoR,
        'noJER' : args.noJER,
        'noJEC' : args.noJEC,
        'noJUNC' : args.noJUNC,
        'noPUweight' : args.noPUweight,
        'noPrefireSF' : args.noPrefireSF,
        'noIDsfs' : args.noIDsfs,
        'noIsosfs' : args.noIsosfs,
        'noTriggersfs' : args.noTriggersfs,
        'noBtagSF' : args.noBtagSF,
        'Zreweight' : args.Zreweight,
        'noBkgVeto' : args.noBkgVeto,
        'verbose' : args.verbose,
        'syst' : args.syst,
        'basepath' : os.path.join(destination, out_fname)
    }

    if args.verbose:
        print("extra args")
        print(json.dumps(argsdict, indent=4))
        print()

    argsdict['config'] = config

    def process_func(inputfiles, args, filesplit):
        final_result = None

        processor_instance = EECProcessor(**args)

        import uproot
        for inputfile in inputfiles:
            nevts = uproot.open(inputfile)['Events'].num_entries
            
            for filesplit_k in range(filesplit):
                #print("processing file %s, filesplit %d/%d"%(inputfile, 
                #                                             filesplit_k,
                #                                             filesplit))
                try:
                    start = filesplit_k*(nevts//filesplit)

                    if filesplit_k == filesplit-1:
                        end = nevts
                    else:
                        end = (filesplit_k+1)*(nevts//filesplit)

                    schema = NanoAODSchema
                    schema.warn_missing_crossrefs = False
                    events = NanoEventsFactory.from_root(
                        inputfile, 
                        entry_start=start,
                        entry_stop=end,
                        schemaclass = schema
                    ).events()

                    nextresult = processor_instance.process(events)
                except:
                    import traceback
                    traceback.print_exc()
                    nextresult = {'errd' : [inputfile]}

                if final_result is None:
                    final_result = nextresult
                else:
                    iadd(final_result, nextresult)
        return final_result
    
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
        if args.verbose:
            print("Running on slurm")
        cluster, client = setup_cluster_on_submit(1, 200, destination)
    elif args.scale == 'local':
        if args.verbose:
            print("Running locally")
        cluster, client = setup_local_cluster(32)
    elif args.scale == 'local_debug':
        if args.verbose:
            print("Running locally in debug mode")
        cluster, client = setup_local_cluster(1)

    ##################################################

    ################### RUNNING ###################

    import dask

    from dask.distributed import as_completed
    from tqdm import tqdm
    from iadd import iadd

    result_futures = None

    #target sumwt is 208328.90519900652

    from tree_acc import tree_acc

    from dask.distributed import progress

    from more_itertools import batched
    inputfiles_l = list(batched(files, args.filebatch))

    #result = tree_acc(client.map(process_func, inputs, args=argsdict), client)
    #progress(result)

    t = tqdm(as_completed(client.map(process_func, inputfiles_l, 
                                     args=argsdict,
                                     filesplit=args.filesplit), 
                          with_results=False,
                          raise_errors=False), 
             total=len(inputfiles_l),
             leave=True,
             miniters=1,
             smoothing=0.1,
             desc='Processing...')
    from accumulatefutures import accumulate_results
    final_ans = accumulate_results(t, 
                                   args.numCollectionThreads,
                                   args.numAddingProcesses,
                                   simple=True)
    #for future in t:
    #    if future.status == 'finished':
    #        if result_futures is None:
    #            result_futures = future
    #        else:
    #            result_futures = \
    #                client.submit(sum_func, result_futures, future,
    #                              priority=999)
    #            
    #            del future
    #        '''
    #        if final_ans is None:
    #            final_ans = future.result()
    #        else:
    #            iadd(final_ans, future.result())
    #        t.set_description("Events: %g"%final_ans['nominal']['sumwt'], 
    #                          refresh=True)
    #        '''
    #    else:
    #        print("Error in processing file")
    #        import traceback
    #        traceback.print_tb(future.traceback())
    #        print(future.exception())
    #        future.release()

    #print("cleaning up collecting threads")
    #for t in collectingThreads:
    #    completedFutures.put(None)
    #    t.join()
    #completedFutures.join()

    #print("cleaning up adding processes")
    #print("\tputting none")
    #for p in addingProcesses:
    #    collectedFutures.put(None)
    #print("closing collectedFutures")
    #collectedFutures.close()
    #print("joining collectedFutures")
    #collectedFutures.join_thread()

    #print("joining adding processes")
    #for p in addingProcesses:
    #    print(p)
    #    #p.terminate()

    #print("waiting an extra 10 seconds to ensure everything is processed")
    #time.sleep(10)

    #print("accumulating final result")
    #final_ans = None
    #while(not resultQ.empty()):
    #    ans = resultQ.get()
    #    print("partial sumw: %g"%ans['nominal']['sumwt'])
    #    if final_ans is None:
    #        final_ans = ans
    #    else:
    #        iadd(final_ans, ans)
    #print("final sumw: %g"%final_ans['nominal']['sumwt'])

    #for p in addingProcesses:
    #    print(p)
    #    p.join()

    #final_ans = result_futures.result()

    print("Done processing.")
    if 'errd' in final_ans:
        print("\t%d jobs failed"%len(final_ans['errd']))
    else:
        print("\tAll jobs computed successfully")

    if args.recover:
        for key in final_ans.keys():
            if key not in ['errd', 'config'] and key in prev.keys():
                iadd(final_ans[key], prev[key])

    with open(os.path.join(destination,out_fname+"_%s.pickle"%args.syst), 'wb') as fout:
        import pickle
        pickle.dump(final_ans, fout)

    if args.verbose:
        print("cleaning up dask cluster")
    client.close()
    cluster.close()
    #####################################################
