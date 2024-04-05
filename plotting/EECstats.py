import numpy as np
import numbers

'''
Implements correct statistical treatments as derived in 
"Statistical uncertainties" section of AN
'''

def diagonal(x):
    if len(x.shape) == 2:
        return np.einsum('ii->i', x, optimize=True)
    elif len(x.shape) == 4:
        return np.einsum('ijij->ij', x, optimize=True)
    else:
        raise ValueError("Expected 2D or 4D array")

def getsum(val1, val2, cov1, cov2, cov1x2=None):
    val1, cov1, shape1 = flatten(val1, cov1)
    val2, cov2, shape2 = flatten(val2, cov2)
    cov1x2 = flatten_cov1x2(cov1x2, shape1, shape2)

    ans = val1+val2

    term1 = cov1
    term2 = cov2
    if cov1x2 is not None:
        term3 = 2*cov1x2
    else:
        term3 = np.zeros_like(term1)

    covans = term1 + term2 + term3

    ans, covans = unflatten(ans, covans, shape1)

    return ans, covans

def getdifference(val1, val2, cov1, cov2, cov1x2=None):
    val1, cov1, shape1 = flatten(val1, cov1)
    val2, cov2, shape2 = flatten(val2, cov2)
    cov1x2 = flatten_cov1x2(cov1x2, shape1, shape2)

    ans = val1-val2

    term1 = cov1
    term2 = cov2
    if cov1x2 is not None:
        term3 = 2*cov1x2
    else:
        term3 = np.zeros_like(term1)

    covans = term1 + term2 - term3

    ans, covans = unflatten(ans, covans, shape1)

    return ans, covans

def getproduct(val1, val2, cov1, cov2, cov1x2=None):
    val1, cov1, shape1 = flatten(val1, cov1)
    val2, cov2, shape2 = flatten(val2, cov2)
    cov1x2 = flatten_cov1x2(cov1x2, shape1, shape2)

    ans = val1*val2

    term1 = np.einsum('i, j, ij -> ij', val2, val2, cov1,
                      optimize=True)
    term2 = np.einsum('i, j, ij -> ij', val1, val1, cov2,
                      optimize=True)
    if cov1x2 is not None:
        term3 = np.einsum('i, j, ij -> ij', val1, val2, cov1x2,
                          optimize=True)
        term4 = np.einsum('i, j, ij -> ij', val2, val1, cov1x2,
                          optimize=True)
    else:
        term3 = np.zeros_like(term1)
        term4 = np.zeros_like(term1)

    covans = term1 + term2 + term3 + term4

    ans, covans = unflatten(ans, covans, shape1)

    return ans, covans

def getratio(val1, val2, cov1, cov2, cov1x2=None):
    val1, cov1, shape1 = flatten(val1, cov1)
    val2, cov2, shape2 = flatten(val2, cov2)
    cov1x2 = flatten_cov1x2(cov1x2, shape1, shape2)

    ans = val1/val2

    inv2 = np.where(val2 > 0, 1/val2, 0)
    sqinv2 = np.square(inv2)
    term1 = np.einsum('i, j, ij -> ij', inv2, inv2, cov1,
                      optimize=True)
    term2 = np.einsum('i, j, i, j, ij -> ij', val1, val1, 
                      sqinv2, sqinv2, cov2, optimize=True)
    if cov1x2 is not None:
        term3 = np.einsum('i, j, i, j, ij -> ij', val1, val2, 
                          sqinv2, sqinv2, cov1x2, optimize=True)
        term4 = np.einsum('i, j, i, j, ij -> ij', val2, val1,
                          sqinv2, sqinv2, cov1x2, optimize=True)
    else:
        term3 = np.zeros_like(term1)
        term4 = np.zeros_like(term1)

    covans = term1 + term2 - term3 - term4

    ans, covans = unflatten(ans, covans, shape1)

    return ans, covans

def maybe_density_cross(vals, ovals, density1, density2, N1, N2,
                        cov1x2):
    '''
    NB expects vals, ovals to already have been normalized
    '''
    vals, ovals, cov1x2, shape1, shape2 = flaten_cross(vals, ovals, cov1x2)

    if cov1x2 is None or (N1==1 and N2==1):
        return cov1x2

    if density1 != density2:
        raise NotImplementedError("Case where exactly one of the distributions is normalized is not supported. Why are you even trying to do this??")

    if type(density1) == bool:
        if(density1):
            term1 = np.einsum('i, ab, j -> ij', vals, cov1x2, ovals,
                              optimize=True)/(N1*N2)
            term2 = (np.einsum('i, aj -> ij', vals, cov1x2,
                               optimize=True)/(N1*N2))
            term3 = (np.einsum('j, ib -> ij', ovals, cov1x2,
                               optimize=True)/(N1*N2))
            term4   = cov1x2/(N1*N2)
            return unflatten_cross(term1 - term2 - term3 + term4, shape1, shape2)
        else:
            return unflatten_cross(cov1x2, shape1, shape2)
    elif isinstance(density1, numbers.Number):
        return unflatten_cross(cov1x2/(N1*N2), shape1, shape2)

def maybe_density(vals, cov, density, return_N=False):
    vals, cov, shape = flatten(vals, cov)

    if len(vals.shape) != 1 or len(cov.shape) != 2:
        raise ValueError("Expected vals to be 1D and cov to be 2D")

    if type(density) is bool:
        if density:
            #use eqn from AN
            N = np.sum(vals)
            term1 = np.einsum('i, ab, j -> ij', vals, cov, vals, 
                              optimize=True)/(N**4)
            term2 = (np.einsum('i, aj -> ij', vals, cov,
                               optimize=True)/(N**3))
            term3 = (np.einsum('j, ib -> ij', vals, cov,
                               optimize=True)/(N**3))
            term4 = cov/(N**2)
            normvals = vals/N
            normcov = term1 - term2 - term3 + term4
        else:
            normvals = vals
            normcov = cov
            N = 1
    elif isinstance(density, numbers.Number):
        #If given a number, it's independent
        normvals = vals/density
        normcov = cov/(density**2)
        N = density

    normvals, normcov = unflatten(normvals, normcov, shape)

    if return_N:
        return normvals, normcov, N
    else:
        return normvals, normcov

def applyRelation(vals, cov, oval, ocov, cov1x2, mode):
    if mode == 'ratio':
        ans, covans = getratio(vals, oval, cov, ocov, cov1x2)
    elif mode == 'difference':
        ans, covans = getdifference(vals, oval, cov, ocov, cov1x2)
    elif mode == 'sigma':
        ans, covans = getdifference(vals, oval, cov, ocov, cov1x2)
        ans = ans/np.sqrt(diagonal(covans))
        covans = covans/diagonal(covans)
    elif mode == 'sum':
        ans, covans = getsum(vals, oval, cov, ocov, cov1x2)
    elif mode == 'product':
        ans, covans = getproduct(vals, oval, cov, ocov, cov1x2)
    return ans, covans

def change_basis(vals, cov, A):
    Ndim = len(cov.shape)//2
    Aindices = [i for i in range(Ndim)] + [i+10 for i in range(Ndim)]
    covindices = [i+10 for i in range(Ndim)] + [i+20 for i in range(Ndim)]
    Aindices2 = [i+20 for i in range(Ndim)] + [i+30 for i in range(Ndim)]
    ansindices = [i for i in range(Ndim)] + [i+30 for i in range(Ndim)]

    covforward = np.einsum(A, Aindices, cov, covindices,  
                           A, Aindices2, ansindices, optimize=True)

    valindices = [i+10 for i in range(Ndim)]
    forwardindices = [i for i in range(Ndim)]
    forward = np.einsum(A, Aindices, vals, valindices,
                        forwardindices, optimize=True)

    return forward, covforward

def flatten(vals, cov):
    shape = vals.shape
    size = np.prod(shape)
    vals = vals.reshape(-1)

    cov = cov.reshape((size, size))
    return vals, cov, shape

def unflatten(vals, cov, shape):
    vals = vals.reshape(shape)
    cov = cov.reshape((*shape, *shape))
    return vals, cov

def flaten_cross(vals1, vals2, cov1x2):
    shape1 = vals1.shape
    shape2 = vals2.shape
    size1 = np.prod(shape1)
    size2 = np.prod(shape2)

    vals1 = vals1.reshape(-1)
    vals2 = vals2.reshape(-1)
    if cov1x2 is not None:
        cov1x2 = cov1x2.reshape((size1, size2))
    return vals1, vals2, cov1x2, shape1, shape2

def unflatten_cross(cov1x2, shape1, shape2):
    cov1x2 = cov1x2.reshape((*shape1, *shape2))
    return cov1x2

def flatten_cov1x2(cov1x2, shape1, shape2):
    if cov1x2 is None:
        return None

    size1 = np.prod(shape1)
    size2 = np.prod(shape2)
    cov1x2 = cov1x2.reshape((size1, size2))
    return cov1x2
