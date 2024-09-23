from iadd import iadd

def add_pair(future1, future2):
    try:
        if future1 is not None and future2 is not None:
            return iadd(future1, future2)
        elif future1 is not None:
            return future1
        elif future2 is not None:
            return future2
        else:
            return None
    except:
        return {'errd' : ['add_pair?']}

def tree_acc(futures, client):
    if len(futures) == 1:
        return futures[0]
    elif len(futures) == 2:
        if client is None:
            return add_pair(futures[0], futures[1])
        else:
            return client.submit(add_pair,
                                 futures[0], 
                                 futures[1])
    else:
        N = len(futures)//2
        if client is None:
            return add_pair(tree_acc(futures[:N], client), 
                            tree_acc(futures[N:], client))
        else:
            return client.submit(add_pair, 
                                 tree_acc(futures[:N], client), 
                                 tree_acc(futures[N:], client))

