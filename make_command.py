
#hadchfilters = ['Tight', 'Loose']
#had0filters = ['Tight', 'Loose']
#elefilters = ['Tight', 'Loose']
#em0filters = ['Tight', 'Loose']

#dropgens = ['Yes', 'No']
#chargefilters = ['Any', 'Tight']
#recover = ["Free", 'Limited', 'No']
#thresholds = ['Yes', 'No']


hadchfilters = ['Loose']
had0filters = ['Tight', 'Loose']
elefilters = ['Loose']
em0filters = ['Loose']

dropgens = ['Yes', 'No']
chargefilters = ['Tight']
recover = ['Limited']
thresholds = ['Yes']

import itertools

names = []

for setting in itertools.product(hadchfilters, had0filters,
                                 elefilters, em0filters,
                                 dropgens, chargefilters, 
                                 recover, thresholds):
    print(setting)
    hadchfilter, had0filter, elefilter, em0filter,\
        dropgen, chargefilter, recover, threshold = setting

    name = '%s%s%s%s%s%s%s%s'%(hadchfilter, had0filter, elefilter, em0filter,
                               dropgen, chargefilter, recover, threshold)
    names += [name]

print(len(names))

for name in names:
    print(name)
