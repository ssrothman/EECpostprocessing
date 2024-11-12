import fsspec_xrootd as xrdfs
import os.path

def get_rootfiles(hostid, path):
    if hostid is None:
        from fsspec.implementations.local import LocalFileSystem
        fs = LocalFileSystem()
    else:
        fs = xrdfs.XRootDFileSystem(hostid = hostid, timeout=60)
    return get_files_recursive(fs, path,
                               lambda f : f.endswith(".root") and not os.path.basename(f).startswith('dropped'), 
                               'root://%s/'%hostid if hostid is not None else '')

def get_files_recursive(fs, rootpath, allowed = lambda f : True, prepend = ''):
    pathlist = fs.ls(rootpath, detail=True)
    result = []
    for path in pathlist:
        if path['type'] == 'directory':
            result += get_files_recursive(fs, path['name'], allowed, prepend)
        elif path['type'] == 'file':
            if allowed(path['name']):
                result.append(prepend + path['name'])
        else:
            raise RuntimeError("Unexpected file type: {}".format(path['type']))
    return result

submitfs = xrdfs.XRootDFileSystem(hostid = "submit54.mit.edu")
