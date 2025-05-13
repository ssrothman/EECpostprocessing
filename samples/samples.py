import os

def get_flags(Hpath):
    path = os.path.basename(Hpath)
    flags = path.split('_')
    flags = list(filter(lambda x: 'file' not in x and 'hist' not in x, flags))
    return flags

def flags_match(Hpath, flags):
    Hflags = get_flags(Hpath)
    print("Hflags:", Hflags)
    print("flags:", flags)
    if len(Hflags) != len(flags):
        print("flag length mismatch")
        return False
    for Hflag in Hflags:
        okay = False
        for flag in flags:
            if flag in Hflag:
                okay = True
                break
        if not okay:
            print("flag %s not found in %s"%(Hflag, flags))
            return False
    print("flags match")
    return True

def no_flags(Hpath):
    flags = get_flags(Hpath)
    return len(flags) == 0

class Sample:
    def __init__(self, name, tag, location, JEC, flags):
        self._name = name
        if type(tag) not in [list, tuple]:
            self._tag = [tag]
        else:
            self._tag = tag

        self._location = location
        self._JEC = JEC
        self._flags = flags

    def get_files(self, exclude_dropped=True):
        from reading.files import get_rootfiles
        result = []
        for t in self._tag:
            if self.location == 'LPC':
                hostid = "cmseos.fnal.gov"
                rootpath = '/store/group/lpcpfnano/srothman/%s'%t
            elif self.location == 'LPC-PERSONAL':
                hostid = "cmseos.fnal.gov"
                rootpath = '/store/user/srothman/%s'%t
            elif self.location == 'MIT':
                hostid = 'submit50.mit.edu'
                rootpath = '/store/user/srothman/%s'%t
            elif self.location == 'scratch':
                hostid = None
                rootpath = '/scratch/submit/cms/srothman/%s'%t
            elif self.location == 'test':
                hostid = None
                rootpath = '/work/submit/srothman/EEC/CMSSW_10_6_26/src/SRothman/%s'%t
            result += get_rootfiles(hostid, rootpath, exclude_dropped=exclude_dropped)
        return result

    @property
    def name(self):
        return self._name

    @property
    def tag(self):
        return self._tag

    @property
    def location(self):
        return self._location

    @property
    def JEC(self):
        return self._JEC

    @property
    def flags(self):
        return self._flags

    @property
    def version(self):
        from pathlib import PurePath
        return PurePath(self.tag).parts[0]

class SampleSet:
    def __init__(self, tag):
        self._samples = {}
        self._tag = tag

    @property
    def tag(self):
        return self._tag

    def add_sample(self, sample):
        self._samples[sample.name] = sample

    def get_samples(self):
        return self._samples.values()

    def lookup(self, name):
        if name in self._samples:
            return self._samples[name]
        else:
            print("Sample %s not found"%name)
            print("available samples:")
            for s in self._samples:
                print("\t",s)

    def get_hist(self, name, what, flags=None):
        import pickle
        import glob
        import os

        if flags is not None and len(flags) == 0:
            flags = None

        basepath = self.get_basepath(name, what)

        options = glob.glob(os.path.join(basepath, "*.pkl"))
        if len(options) == 0:
            print("WARNING: no histograms found")
            print("basepath:", basepath)
            return
            
        print("OPTIONS")
        print(options)
        print("FLAGS")
        print(flags)
        if flags is not None:
            options = list(filter(lambda x: flags_match(x, flags), options))
        else:
            options = list(filter(no_flags, options))

        print("FILTERED")
        print(options)
        print()
    
        #choose most recently edited
        histpath = max(options, key=os.path.getmtime)

        if len(options) > 1:
            print("WARNING: multiple available histograms")
            print("chosing most recent:", histpath)

        print(histpath)
        with open(histpath, 'rb') as f:
            return pickle.load(f) 

    def get_basepath(self, name, what):
        return os.path.join("/data/submit/srothman/EEC", 
                            self.tag, 
                            name,
                            what)



