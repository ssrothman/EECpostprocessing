import plotting.EECutil as EECutil
import plotting.plotEEC as plotEEC
import matplotlib.pyplot as plt

x = EECutil.EEChistReader('Feb05_2024_highstats/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7/Herwig/EEC/hists.pkl')

bins = {'order' : 0, 'pt' : 2}
#plotEEC.plotEEC(x, 'EECs', 'Hreco', density = False, bins=bins)
#plotEEC.plotEEC(x, 'EECs', 'HrecoUNMATCH', density = False, bins=bins)
#plt.show()
plotEEC.plotRatio(x, 'EECs', 'Hreco', False, 
                  x, 'ChargedEECs', 'Hreco', False,
                  bins1=bins, bins2=bins,
                  mode='sigma')
plt.show()
