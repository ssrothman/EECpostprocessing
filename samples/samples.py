import os

class Sample:
    def __init__(self, name, tag, location, JEC, flags):
        self._name = name
        self._tag = tag
        self._location = location
        self._JEC = JEC
        self._flags = flags

    def get_files(self):
        from reading.files import get_rootfiles
        if self.location == 'LPC':
            hostid = "cmseos.fnal.gov"
            rootpath = '/store/group/lpcpfnano/srothman/%s'%self.tag
        elif self.location == 'MIT':
            hostid = 'submit50.mit.edu'
            rootpath = '/store/user/srothman/%s'%self.tag
        return get_rootfiles(hostid, rootpath)

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

    def get_basepath(self):
        return os.path.join("/data/submit/srothman/EEC", 
                            self.version, 
                            self.name)

    def get_hist(self, what, flags=None):
        import pickle
        import glob
        import os

        if flags is not None and len(flags) == 0:
            flags = None

        basepath = os.path.join("/data/submit/srothman/EEC", 
                                self.version, 
                                self.name,
                                what)

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
            for flag in flags:
                options = list(filter(lambda x: flag in x, options))
        else:
            options = list(filter(lambda x: os.path.basename(x).count('_')==0 or ('_file' in x and os.path.basename(x).count("_")==1),
                           options))
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
