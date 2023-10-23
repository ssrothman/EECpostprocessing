#thanks to https://stackoverflow.com/questions/6931342/system-wide-mutex-in-python-on-linux
import fcntl

class Locker:
    def __enter__ (self):
        self.fp = open("./lockfile.lck", 'r+')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()
