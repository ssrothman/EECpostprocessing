import multiprocessing
import queue 
import threading

def handle_io(inQ, outQ, sumwtQ, ithread):
    #print("starting collector",ithread,flush=True)
    while True:
        future = inQ.get()
        #print("collector",ithread,"got future",flush=True)
        if future is None:
            #print("terminating collector",ithread,flush=True)
            break

        theresult = future.result()
        outQ.put(theresult)
        sumwtQ.put(theresult['nominal']['sumwt'])

        future.release()
        #print("collector",ithread,"put result",flush=True)
    outQ.put(None)
    #print("collector",ithread,"DONE",flush=True)

def handle_accumulation(inQ, outQ, iproc):
    #print("starting accumulator",iproc,flush=True)
    from iadd import iadd
    ans = None
    while True:
        result = inQ.get()
        #print("accumulator",iproc,"got result",flush=True)
        if result is None:
            #print("terminating accumulator",iproc,flush=True)
            break

        if ans is None:
            ans = result
        else:
            iadd(ans, result)
        #print("accumulator",iproc,"added result",flush=True)

    outQ.put(ans)
    #print("accumulator",iproc,"done",flush=True)

def accumulate_results(ascompletediterable,
                       num_collecting_threads=4,
                       num_accumulating_processes=4,
                       joinTimeout=30,
                       use_processes=False,
                       simple=True):
    '''
    Use multithreading and multiprocessing to accumulate future results 
    in the background. The rationale is as follows:

     - For memory efficiency we want to release futures as quickly as possible
     - For speed we want to accumulate results as quickly as possible

    Standard as_completed(with_results=False) will block 
    for each future's I/O and accumulation, which is very inefficient

    Standard as_completed(with_results=True) is better, 
    but has some pathalogical edge cases. 
    In particular, it yeilds results based on when the I/O STARTED, 
    not based on when the I/O COMPLETES.
    This means that if the I/O for a given future takes a long time 
    (or worse encounters and error and hangs)
    then the whole loop will block until that is resolved.
    In the meantime, the I/O for other completeing futures is still happening
    in the background, but the results are not being accumulated.
    This means that time is being wasted,
    and even worse the memory consumption can blow up.

    Instead, we pass the as_completed iterator to this function, 
    which drops the results into a queue as they are ready, 
    but does not perform any I/O (or any other blocking operations)

    This queue is processed by a set of /threads/, which are necessary 
    because the futures are shared memory objects, and if you serialize them
    then you lose the ability to actually call future.result()
    The GIL is not a problem anyway, as just we're doing network I/O

    These threads drop the collected results into another queue, 
    which is then processed by multiprocessing processes to avoid the GIL.

    @param ascompletediterable: an as_completed iterator
    @param num_collecting_threads: number of threads to use 
                                   to collect the results (ie network I/O)
    @param num_accumulating_processes: number of processes to use 
                                       to accumulate the results (ie iadd(a,b))

    @return: the accumulated results
    '''

    if not simple:
        #setup queues
        completed_futures = queue.Queue()
        if use_processes:
            collected_futures = multiprocessing.Queue()
            accumulated_results = multiprocessing.Queue()
            sumwt_queue = multiprocessing.Queue()
            total_sumwt = {'sumwt' : 0}
        else:
            collected_futures = queue.Queue()
            accumulated_results = queue.Queue()
            sumwt_queue = queue.Queue()
            total_sumwt = {'sumwt' : 0}

        #setup collecting threads
        collecting_threads = [
            threading.Thread(target=handle_io, 
                             args=(completed_futures,
                                   collected_futures, 
                                   sumwt_queue,
                                   i),
                             daemon=True)
            for i in range(num_collecting_threads)
        ]
        for t in collecting_threads:
            t.start()

        #setup accumulating processes
        if not use_processes:
            accumulating_processes = [
                threading.Thread(target=handle_accumulation,
                                 args=(collected_futures, accumulated_results, i),
                                 daemon=True)
                for i in range(num_accumulating_processes)
            ]
        else:
            accumulating_processes = [
                multiprocessing.Process(target=handle_accumulation,
                                       args=(collected_futures, 
                                             accumulated_results, i),
                                        daemon=True)
                for i in range(num_accumulating_processes)
            ]
        for p in accumulating_processes:
            p.start()

        #setup sumwt monitoring thread
        def accumulate_sumwt(q, total):
            thesumwt = 0
            #print("starting sumwt process")
            while True:
                sumwt = q.get()
                #print("got sumwt",sumwt)
                if sumwt is None:
                    break
                thesumwt += sumwt
                total['sumwt'] = thesumwt
                #print("sumwt",thesumwt)
                #print()

        sumwt_thread = threading.Thread(target=accumulate_sumwt,
                                        args=(sumwt_queue, total_sumwt),
                                        daemon=True)
        sumwt_thread.start()

    if simple:
        final_ans = None

    #feed futures to collecting threads
    for future in ascompletediterable:
        if future.status == 'finished':
            if not simple:
                completed_futures.put(future)
            else:
                if final_ans is None:
                    final_ans = future.result()
                else:
                    from iadd import iadd
                    iadd(final_ans, future.result())
                future.release()
        elif future.status == 'error':
            print("error in future",future,flush=True)
            import traceback
            traceback.print_tb(future.traceback())
            print(future.exception())
            future.release()
        else:
            print("WARNING: unexpected future status",future.status,flush=True)
            future.release()

        if not simple:
            ascompletediterable.set_description(
                "Events: %g"%total_sumwt['sumwt'],
                refresh=True
            )

    if not simple:
        #send stop signals
        for i in range(num_collecting_threads):
            completed_futures.put(None)

        for i in range(num_accumulating_processes):
            collected_futures.put(None)

        #wait for collecting threads to finish
        for t in collecting_threads:
            t.join(joinTimeout)

        #wait for accumulating processes to finish
        for p in accumulating_processes:
            p.join(joinTimeout)

        for t in collecting_threads:
            if t.is_alive():
                print("WARNING: collecting thread",t,"did not terminate",flush=True)
                #t.terminate()

        for p in accumulating_processes:
            if p.is_alive():
                print("WARNING: accumulating process",p,"did not terminate",flush=True)
                #p.terminate()

        #collect results
        from iadd import iadd
        final_ans = None
        while not accumulated_results.empty():
            result = accumulated_results.get()
            if final_ans is None:
                final_ans = result
            else:
                iadd(final_ans, result)

    return final_ans
