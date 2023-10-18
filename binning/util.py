import hist

def getLogAxis(name='pt', label='$p_T$ [GeV]', bins=50, minval=0.1, maxval=100):
    return hist.axis.Regular(bins, minval, maxval, name=name, label=label, transform=hist.axis.transform.log)

def getVariableAxis(name='eta', label='$|\eta|$', vals=[0.0, 3.0]):
    return hist.axis.Variable(vals, name=name, label=label, overflow=False, underflow=False)

def getLinAxis(name='dpt', label='$\Delta p_T$ [%]', bins=50, minval=-100, maxval=100):
    return hist.axis.Regular(bins, minval, maxval, name=name, label=label)

def getCatAxis(name='pdgid', label='Particle ID', cats=[11, 13, 22, 130, 211]):
    return hist.axis.IntCategory(cats, name=name, label=label)

def getIntAxis(name='nmatch', label='Number of Matches', minval=0, maxval=3):
    return hist.axis.Integer(minval, maxval, name=name, label=label)
