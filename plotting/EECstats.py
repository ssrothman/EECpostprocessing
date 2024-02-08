import numpy as np
import numbers

'''
Implements correct statistical treatments as derived in 
"Statistical uncertainties" section of AN
'''

def getsum(val1, val2, cov1, cov2, cov1x2=None):
    ans = val1+val2

    term1 = cov1
    term2 = cov2
    if cov1x2 is not None:
        term3 = 2*cov1x2
    else:
        term3 = np.zeros_like(term1)

    return ans, term1 + term2 + term3

def getdifference(val1, val2, cov1, cov2, cov1x2=None):
    ans = val1-val2

    term1 = cov1
    term2 = cov2
    if cov1x2 is not None:
        term3 = 2*cov1x2
    else:
        term3 = np.zeros_like(term1)

    return ans, term1 + term2 - term3

def getproduct(val1, val2, cov1, cov2, cov1x2=None):
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

    return ans, term1 + term2 + term3 + term4

def getratio(val1, val2, cov1, cov2, cov1x2=None):
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

    return ans, covans

def maybe_density(vals, cov, density):
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
    elif isinstance(density, numbers.Number):
        #If given a number, it's independent
        normvals = vals/density
        normcov = cov/(density**2)

    return normvals, normcov

def applyRelation(vals, cov, oval, ocov, cov1x2, mode):
    if len(cov.shape) == 1:
        cov = np.diag(cov)
    if len(ocov.shape) == 1:
        ocov = np.diag(ocov)

    if mode == 'ratio':
        ans, covans = getratio(vals, oval, cov, ocov, cov1x2)
    elif mode == 'difference':
        ans, covans = getdifference(vals, oval, cov, ocov, cov1x2)
    elif mode == 'sigma':
        ans, covans = getdifference(vals, oval, cov, ocov, cov1x2)
        ans = ans/np.sqrt(np.einsum('ii->i',covans, optimize=True))
        covans = covans/np.einsum('ii->i',covans, optimize=True)
    elif mode == 'sum':
        ans, covans = getsum(vals, oval, cov, ocov, cov1x2)
    elif mode == 'product':
        ans, covans = getproduct(vals, oval, cov, ocov, cov1x2)
    return ans, covans

