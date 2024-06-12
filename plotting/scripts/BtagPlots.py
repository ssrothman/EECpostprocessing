from plotting.util import *
from plotting.plotBtag import *
from samples.latest import SAMPLE_LIST

import matplotlib.pyplot as plt

tight = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Btag", ['tight',
                                                                 'genXsec'])
medium = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Btag", ['medium',
                                                                  'genXsec'])
loose = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Btag", ['loose',
                                                                 'genXsec'])

plotBmatch(tight)

plotBtagPurePerPt(tight, 'tight', 0, 'pass')
plotBtagPurePerPt(tight, 'tight', 1, 'pass')

plotBtagPurePerPt(tight, 'tight', 0, 'fail')
plotBtagPurePerPt(tight, 'tight', 1, 'fail')

#plotBtagPurePerPt(medium, 'medium', 0, 'pass')
#plotBtagPurePerPt(loose, 'loose', 0, 'pass')

#plotBtagPurePerPt(medium, 'medium', 1, 'pass')
#plotBtagPurePerPt(loose, 'loose', 1, 'pass')

plotBtagEffPerPt(tight, 'tight', 0, 'udsg', mode='pass')
plotBtagEffPerPt(tight, 'tight', 1, 'c', mode='pass')
plotBtagEffPerPt(tight, 'tight', 2, 'b', mode='pass')

plotBtagEffPerPt(medium, 'medium', 0, 'udsg', mode='pass')
plotBtagEffPerPt(medium, 'medium', 1, 'c', mode='pass')
plotBtagEffPerPt(medium, 'medium', 2, 'b', mode='pass')

plotBtagEffPerPt(loose, 'loose', 0, 'udsg', mode='pass')
plotBtagEffPerPt(loose, 'loose', 1, 'c', mode='pass')
plotBtagEffPerPt(loose, 'loose', 2, 'b', mode='pass')

plotBtag(tight, 'purity')
plotBtag(tight, 'efficiency')

plotBtag(medium, 'purity')
plotBtag(medium, 'efficiency')

plotBtag(loose, 'purity')
plotBtag(loose, 'efficiency')
