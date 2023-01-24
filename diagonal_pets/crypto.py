# Classes that maintain encrypted counts of the total number of visits
# made to places, and the number of visits made by people infected on
# a given day. Counts are effectively stored in large arrays directly
# indexed by place ID, broken up into groups of GROUP_SIZE. Groups are
# encrypted with the homomorphic BFV scheme.
# We expose methods to add encrypted counts, and to lookup the encrypted value
# of counts associated with given indices.

import concurrent.futures
import gzip
import io
import numpy as np
import os
import random
import struct
import threading

GROUP_SIZE = 2048
DTYPE = np.int64
SCHEME = "BFV"

def make_pyfhel(fake):
    if fake:
        return FakePyfhel(), FakePyCtxt
    else:
        import Pyfhel
        return Pyfhel.Pyfhel(), Pyfhel.PyCtxt

def init_pyfhel(h):
    # We allocate twice the number of slots as GROUP_SIZE, as we need to
    # be able to rotate a count at index GROUP_SIZE-1 to index 0 during
    # lookup, and rotations are only valid up to n/2.
    h.contextGen(scheme=SCHEME, n=2*GROUP_SIZE, t_bits=20, sec=128)

# EncryptedLookup represents the encrypted results of looking up counts
# by index within EncryptedCounts
class EncryptedLookup:

    def __init__(self, found, h, ctxt):
        self.found = found
        self.h = h
        self.ctxt = ctxt

    # Shuffle the lookup results according to the given pattern, which
    # specifies the new index for each lookup result.
    def shuffle(self, pattern):
        self.found = [self.found[i] for i in pattern]

    def decrypt(self):
        return [self.h.decryptInt(f)[0] for f in self.found]

    def __len__(self):
        return len(self.found)

# EncryptedCounts represented a set of encrypted counts indexed by place ID
class EncryptedCounts:

    def __init__(self, h, ctxt):
        self.h = h
        self.ctxt = ctxt
        self.groups = []

    @classmethod
    def encrypt(cls, counts, h, ctxt):
        e = EncryptedCounts(h, ctxt)
        for i in range(0, len(counts), GROUP_SIZE):
            group = np.zeros(GROUP_SIZE, dtype=DTYPE)
            for j in range(0, min(GROUP_SIZE, len(counts) - i)):
                group[j] = counts[i+j]
            e.groups.append(h.encryptPtxt(h.encodeInt(group)))
        return e

    def decrypt(self):
        return np.concatenate([self.h.decryptInt(g)[0:GROUP_SIZE] for g in self.groups])

    def add(self, other):
        if not isinstance(other, EncryptedCounts):
            raise TypeError("Expected EncryptedCounts, found %s" % other)
        if not self.groups:
            self.groups = other.groups.copy()
        else:
            for (g1, g2) in zip(self.groups, other.groups):
                self.h.add(g1, g2)

    def clear(self):
        self.groups = []

    # Return the counts associated with each index within the group, returning
    # an encrypted result. Lookup works by finding the group containing the
    # given index, rotating the desired value into index 0, and masking the
    # unwanted values by multiplying all other indices by 0.
    def lookup(self, indicies):
        if self.is_empty():
            raise KeyError("Lookup on empty EncryptedCounts")
        found = []
        if len(indicies) > 0 and self.groups:
            for i in range(0, len(indicies)):
                group = int(indicies[i] / GROUP_SIZE)
                g = self.groups[group].copy()
                self.h.rotate(g, indicies[i] % GROUP_SIZE)
                mask = np.zeros(self.h.n, dtype=DTYPE)
                mask[0] = 1
                self.h.multiply_plain(g, self.h.encodeInt(mask))
                found.append(g)
        return EncryptedLookup(found, self.h, self.ctxt)

    def to_file(self, f):
        cf = gzip.open(f, "wb")
        cf.write(struct.pack("<l", len(self.groups)))
        for c in self.groups:
            b = c.to_bytes()
            cf.write(struct.pack("<l", len(b)))
            cf.write(b)
        cf.close()

    def from_file(self, f):
        self.groups = []
        cf = gzip.open(f, "rb")
        (n,) = struct.unpack("<l", cf.read(4))
        for i in range(0, n):
            (l,) = struct.unpack("<l", cf.read(4))
            self.groups.append(self.ctxt(bytestring=cf.read(l), pyfhel=self.h))
        cf.close()

    def to_bytes(self):
        b = io.BytesIO()
        self.to_file(b)
        return b.getvalue()

    def from_bytes(self, b):
        self.from_file(io.BytesIO(b))

    def is_empty(self):
        return len(self.groups) == 0

class EncryptedAggregatorLookup:

    def __init__(self, visits, infected_visits):
        self.visits = visits
        self.infected_visits = infected_visits

    def decrypt(self):
        return (self.visits.decrypt(), [v.decrypt() for v in self.infected_visits])

# Aggregator represents the total visit counts for each place, and the number
# of visits made by people infected on a given day. Each set of counts is
# represented by an EncryptedCounts instance. As the memory used by
# EncryptedCounts for a large number of places can be significant, we provide
# a functionality to move the counts of visits by people infected on given
# days to and from disk.
class Aggregator:

    DAYS = 64
    FL_PARAMETER_TYPE = "diagonal_pets.Aggregator"

    def __init__(self, h, ctxt):
        self.h = h
        self.ctxt = ctxt
        self.visits = EncryptedCounts(h, ctxt)
        self.infected_visits = [EncryptedCounts(h, ctxt) for i in range(0, self.DAYS)]
        self.day_reference_counts = [0] * self.DAYS
        self.loader = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.preload = None
        self.lock = threading.Lock()

    def encrypt_counts(self, counts):
        return EncryptedCounts.encrypt(counts, self.h, self.ctxt)

    def add_visits(self, counts):
        self.visits.add(counts)

    def add_infected_visits(self, day, counts):
        self.infected_visits[day].add(counts)

    def clear_infected_visits(self, start_day, stop_day):
        for day in range(start_day, stop_day):
            self.infected_visits[day].clear()

    # Lookup the encrypted visit counts (both total and infected for the given
    # range of days) for the given set of places. The returned counts are
    # shuffled, such that their relationship with indicies isn't obvious. While
    # this adds minimal security (since historical visit counts provide a
    # signature that will identify many places), it makes it clear to the
    # caller that reversing this mapping isn't appropriate.
    def lookup(self, indicies, days):
        visits = self.visits.lookup(indicies)
        infected_visits = [self.infected_visits[day].lookup(indicies) for day in days]
        pattern = list(range(0, len(visits)))
        random.shuffle(pattern)
        visits.shuffle(pattern)
        for day in range(0, len(days)):
            infected_visits[day].shuffle(pattern)
        return EncryptedAggregatorLookup(visits, infected_visits)

    def all_visits(self):
        return self.visits

    def all_infected_visits(self, day):
        return self.infected_visits[day]

    def write(self, directory, visits=False, day=-1):
        os.makedirs(directory, exist_ok=True)
        if visits or day < 0:
            with open(directory / "visits", "wb") as f:
                self.visits.to_file(f)
        if not visits:
            days = [day] if day >= 0 else range(0, self.DAYS)
            for day in days:
                with open(directory / ("day-%d" % day), "wb") as f:
                    self.infected_visits[day].to_file(f)

    def read(self, directory, visits=False, day=-1):
        if visits or day < 0:
            with open(directory / "visits", "rb") as f:
                self.visits.from_file(f)
        if not visits:
            days = [day] if day >= 0 else range(0, self.DAYS)
            for day in days:
                self._read_day(directory, day)

    # Increase the reference count for the given range of days. Days without
    # data and a reference count larger than 0 are loaded from disk.
    def reference_window(self, directory, start_day, stop_day):
        self.lock.acquire()
        for day in range(start_day, stop_day):
            self.day_reference_counts[day] += 1
            if self.day_reference_counts[day] > 0 and self.infected_visits[day].is_empty():
                self._read_day(directory, day)
        self.lock.release()

    # Decrease the reference count for the given range of days. Days with
    # data and a reference count of 0 are forgotten.
    def unreference_window(self, start_day, stop_day):
        self.lock.acquire()
        for day in range(start_day, stop_day):
            if self.day_reference_counts[day] > 0:
                self.day_reference_counts[day] -= 1
                if self.day_reference_counts[day] == 0:
                    self.infected_visits[day].clear()
        self.lock.release()

    # Unreference old days, and reference new days, loading and forgetting
    # days as appropriate. As a common case optimisation, start loading
    # the data for 'end' with a separate thread in the background,
    # ensuring it completes by the next call.
    def swap_window(self, directory, old_start, old_end, start, end):
        self.lock.acquire()
        if self.preload is not None:
            concurrent.futures.wait([self.preload])
            e = self.preload.exception()
            if e is not None:
                raise e
            self.preload = None
        for day in range(old_start, old_end):
            if self.day_reference_counts[day] > 0:
                self.day_reference_counts[day] -= 1
        for day in range(start, end):
            self.day_reference_counts[day] += 1
        for (day, count) in enumerate(self.day_reference_counts):
            if count == 0:
                if not self.infected_visits[day].is_empty():
                    self.infected_visits[day].clear()
            elif self.infected_visits[day].is_empty():
                self._read_day(directory, day)
        if end < self.DAYS and self.infected_visits[end].is_empty():
            self.preload = self.loader.submit(self._read_day, directory, end)
        self.lock.release()

    def read_infected_visits_window(self, directory, start_day, stop_day):
        self.lock.acquire()
        for day in range(0, start_day):
            self.infected_visits[day].clear()
        for day in range(stop_day, len(self.infected_visits)):
            self.infected_visits[day].clear()
        for day in range(start_day, stop_day):
            if self.infected_visits[day].is_empty():
                self._read_day(directory, day)
        self.lock.release()

    def _read_day(self, directory, day):
        filename = directory / ("day-%d" % day)
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.infected_visits[day].from_file(f)

    # Fill the Flower parameters instance with a serialised version
    # of the given visit counts.
    def fill_fl_parameters(self, p, visits=False, day=-1):
        if (not visits and day < 0) or (visits and day >=0):
            raise ValueError("Must specify exactly one of visits or day")
        p.tensor_type = self.FL_PARAMETER_TYPE
        if visits:
            p.tensors = [self.all_visits().to_bytes()]
        else:
            p.tensors = [self.all_infected_visits(day).to_bytes()]

    # Add the visit counts serialised into the given Flower parameters.
    def apply_fl_parameters(self, p, visits=False, day=-1):
        if (not visits and day < 0) or (visits and day >=0):
            raise ValueError("Must specify exactly one of visits or day")
        if p.tensor_type != self.FL_PARAMETER_TYPE:
            raise ValueError("Expected tensor_type %s, found %s" % (self.FL_PARAMETER_TYPE, p.tensor_type))
        counts = EncryptedCounts(self.h, self.ctxt)
        counts.from_bytes(p.tensors[0])
        if visits:
            self.add_visits(counts)
        else:
            self.add_infected_visits(day, counts)

def make_aggregator(h, ctxt):
    return Aggregator(h, ctxt)

def write_keys(data, h, secret=True, public=True, rotate=True):
    if secret:
        with open(data.secret_key_filename(), "wb") as f:
            f.write(h.to_bytes_secret_key())
    if public:
        with open(data.public_key_filename(), "wb") as f:
            f.write(h.to_bytes_public_key())
    if rotate:
        with open(data.rotate_key_filename(), "wb") as f:
            f.write(h.to_bytes_rotate_key())

def read_keys(data, h, secret=True, public=True, rotate=True):
    if secret:
        with open(data.secret_key_filename(), "rb") as f:
            h.from_bytes_secret_key(f.read())
    if public:
        with open(data.public_key_filename(), "rb") as f:
            h.from_bytes_public_key(f.read())
    if rotate:
        with open(data.rotate_key_filename(), "rb") as f:
            h.from_bytes_rotate_key(f.read())

# A fake implementation of a subset of the homomorphic crypto
# routines from Pyfhel, for testing in environments where
# Pyfhel isn't present. Operations are performed in the clear.
class FakePyfhel:

    SECRET_KEY = b"secret"
    PUBLIC_KEY = b"public"
    ROTATE_KEY = b"rotate"

    def __init__(self):
        self.secret_key = None
        self.public_key = None
        self.rotate_key = None

    def contextGen(self, **args):
        self.n = args["n"]
        self.secret_key = self.SECRET_KEY
        self.public_key = self.PUBLIC_KEY

    def keyGen(self):
        self.secret_key = self.SECRET_KEY
        self.public_key = self.PUBLIC_KEY

    def rotateKeyGen(self):
        self.rotate_key = self.ROTATE_KEY

    def encodeInt(self, v):
        if len(v) > self.n:
            raise ValueError("Length %d is beyond maximum of %d" % (len(v), self.n))
        return FakePyPtxt(np.pad(v, (0, self.n - len(v))))

    def encryptPtxt(self, p):
        if self.public_key is None:
            raise ValueError("No public key")
        if not isinstance(p, FakePyPtxt):
            raise TypeError("Expected FakePyPtxt")
        return FakePyCtxt(p.v)

    def decryptInt(self, c):
        if self.secret_key is None:
            raise ValueError("No secret key")
        if not isinstance(c, FakePyCtxt):
            raise TypeError("Expected PyCtxt")
        return c.v

    def add(self, c1, c2):
        if not isinstance(c1, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        if not isinstance(c2, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        if self.public_key is None:
            raise ValueError("No public key")
        c1.v += c2.v

    def add_plain(self, c, p):
        if not isinstance(c, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        if not isinstance(p, FakePyPtxt):
            raise TypeError("Expected FakePyPtxt")
        if self.public_key is None:
            raise ValueError("No public key")
        c.v += p.v

    def multiply_plain(self, c, p):
        if not isinstance(c, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        if not isinstance(p, FakePyPtxt):
            raise TypeError("Expected FakePyPtxt")
        if self.public_key is None:
            raise ValueError("No public key")
        c.v *= p.v

    def rotate(self, c, steps):
        if not isinstance(c, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        if abs(steps) >= self.n / 2:
            raise ValueError("Rotate steps %d beyond maximum of %d" % (steps, self.n/2-1))
        if self.rotate_key is None:
            raise ValueError("No rotate key")
        c.v = np.roll(c.v, -steps)

    def to_bytes_secret_key(self):
        return self.SECRET_KEY

    def from_bytes_secret_key(self, b):
        if b != self.SECRET_KEY:
            raise ValueError("Bad secret key")
        self.secret_key = b

    def to_bytes_public_key(self):
        return self.PUBLIC_KEY

    def from_bytes_public_key(self, b):
        if b != self.PUBLIC_KEY:
            raise ValueError("Bad public key")
        self.public_key = b

    def to_bytes_rotate_key(self):
        return self.ROTATE_KEY

    def from_bytes_rotate_key(self, b):
        if b != self.ROTATE_KEY:
            raise ValueError("Bad rotate key")
        self.rotate_key = b

class FakePyPtxt:

    def __init__(self, v):
        self.v = v

class FakePyCtxt:

    MAGIC = b"\xe4\xd4\xd8\xd2"

    def __init__(self, v=None, bytestring=None, pyfhel=None):
        if v is not None:
            self.v = v
        elif bytestring is not None:
            if pyfhel is None:
                raise ValueError("Must pass pyhfel argument with bytestring")
            self.from_bytes(bytestring)
        else:
            raise ValueError("Expected an initial value")

    def to_bytes(self):
        f = io.BytesIO()
        f.write(self.MAGIC)
        for c in self.v:
            f.write(struct.pack("Q", c))
        f.write(self.MAGIC)
        return f.getvalue()

    def from_bytes(self, b):
        if b[0:len(self.MAGIC)] != self.MAGIC:
            raise ValueError("Bad header")
        if b[-len(self.MAGIC):] != self.MAGIC:
            raise ValueError("Bad trailer")
        b = b[len(self.MAGIC):-len(self.MAGIC)]
        self.v = np.zeros(int(len(b)/8), dtype=np.int64)
        for i in range(0, int(len(b)/8)):
            self.v[i] = struct.unpack("Q", b[i*8:(i+1)*8])[0]

    def copy(self):
        return FakePyCtxt(np.copy(self.v))

    def __str__(self):
        return "<encrypted %s>" % (self.v, )

