import hist
import plotting.EECutil
import plotting.plotEEC
import matplotlib.pyplot as plt

path = './Feb28_2024_wcontrol_highstats/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Pythia_ak8/EEC_btag_tight'

x = plotting.EECutil.EEChistReader(path)

base_bins = {'order' : 2, 'pt' : 2}

bins_btag = base_bins.copy()
bins_btag['tag'] = 1

bins_ltag = base_bins.copy()
bins_ltag['tag'] = 0

bins_bgen = base_bins.copy()
bins_bgen['genflav'] = 2

bins_lgen = base_bins.copy()
bins_lgen['genflav'] = 0

print(bins_btag)
print(x.Hdict['EECs']['Hreco'])
print(x.Hdict['EECs']['Hreco'].project("tag"))

plotting.plotEEC.compareEEC_perBins(x, 'EECs', 'Hreco',
                                    ['pass B', 'gen B'],
                                    [bins_btag, bins_bgen],
                                    True)

plotting.plotEEC.compareEEC_perBins(x, 'EECs', 'Hreco',
                                    ['pass B', 'fail B'],
                                    [bins_btag, bins_ltag],
                                    True)

#names = ['l', 'c', 'b']
#for i in range(3):
#    n = names[i]
#    plotting.plotEEC.compareEEC([xgenflav, xboth, xboth, xboth],
#                                ['EECs']*4,
#                                ['Hreco']*4,
#                                ['gen-'+n, 'tag-'+n, 'well-tagged '+n, 'mis-tagged '+n],
#                                [bins_l_genflav[i], bins_l_tag[i], bins_l_correcttag[i], bins_l_wrongtag[i]],
#                                [True]*4)

#plotting.plotEEC.compareEEC([xgenflav, xtag, xtag], 
#                            ['EECs']*3,
#                            ['Hreco']*3, 
#                            ['gen-l', 'l-tagged', '"pure" l'],
#                            [bins_l_genflav[0], bins_l_tag[0], 
#                                            bins_l_pureflav[0]], 
#                            [True]*3)
#plotting.plotEEC.compareEEC([xgenflav, xtag, xtag], 
#                            ['EECs']*3,
#                            ['Hreco']*3, 
#                            ['gen-c', 'c-tagged', '"pure" c'],
#                            [bins_l_genflav[1], bins_l_tag[1], 
#                                            bins_l_pureflav[1]], 
#                            [True]*3)
#plotting.plotEEC.compareEEC([xgenflav, xtag, xtag], 
#                            ['EECs']*3,
#                            ['Hreco']*3, 
#                            ['gen-b', 'b-tagged', '"pure" b'],
#                            [bins_l_genflav[2], bins_l_tag[2], 
#                                            bins_l_pureflav[2]], 
#                            [True]*3)


labels = ['light', 'charm', 'bottom']

#plotting.plotEEC.compareEEC_perBins(xgenflav, 'EECs', 'Hreco', labels,
#                                    bins_l_genflav, True)
#plotting.plotEEC.compareEEC_perBins(xtag, 'EECs', 'Hreco', labels,
#                                    bins_l_tag, True)
#plotting.plotEEC.compareEEC_perBins(xtag, 'EECs', 'Hreco', labels,
#                                    bins_l_pureflav, True)
