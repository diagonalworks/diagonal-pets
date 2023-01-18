import io
import os
import numpy as np
import random
import struct

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
    h.contextGen(scheme=SCHEME, n=2*GROUP_SIZE, t_bits=20, sec=128)

class EncryptedLookup:

    def __init__(self, found, h, ctxt):
        self.found = found
        self.h = h
        self.ctxt = ctxt

    def shuffle(self, pattern):
        self.found = [self.found[i] for i in pattern]

    def decrypt(self):
        return [self.h.decryptInt(f)[0] for f in self.found]

    def __len__(self):
        return len(self.found)

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

    def lookup(self, indicies):
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
        f.write(struct.pack("<l", len(self.groups)))
        for c in self.groups:
            b = c.to_bytes()
            f.write(struct.pack("<l", len(b)))
            f.write(b)

    def from_file(self, f):
        self.groups = []
        (n,) = struct.unpack("<l", f.read(4))
        for i in range(0, n):
            (l,) = struct.unpack("<l", f.read(4))
            self.groups.append(self.ctxt(bytestring=f.read(l), pyfhel=self.h))

    def to_bytes(self):
        b = io.BytesIO()
        self.to_file(b)
        return b.getvalue()

    def from_bytes(self, b):
        self.from_file(io.BytesIO(b))

class EncryptedAggregatorLookup:

    def __init__(self, visits, infected_visits):
        self.visits = visits
        self.infected_visits = infected_visits

    def decrypt(self):
        return (self.visits.decrypt(), [v.decrypt() for v in self.infected_visits])

class Aggregator:

    DAYS = 64
    FL_PARAMETER_TYPE = "diagonal_pets.Aggregator"

    def __init__(self, h, ctxt):
        self.h = h
        self.ctxt = ctxt
        self.visits = EncryptedCounts(h, ctxt)
        self.infected_visits = [EncryptedCounts(h, ctxt) for i in range(0, self.DAYS)]

    def encrypt_counts(self, counts):
        return EncryptedCounts.encrypt(counts, self.h, self.ctxt)

    def add_visits(self, counts):
        self.visits.add(counts)

    def add_infected_visits(self, day, counts):
        self.infected_visits[day].add(counts)

    def clear_infected_visits(self, start_day, stop_day):
        for day in range(start_day, stop_day):
            self.infected_visits[day].clear()

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
                with open(directory / ("day-%d" % day), "rb") as f:
                    self.infected_visits[day].from_file(f)

    def fill_fl_parameters(self, p, visits=False, day=-1):
        if (not visits and day < 0) or (visits and day >=0):
            raise ValueError("Must specify exactly one of visits or day")
        p.tensor_type = self.FL_PARAMETER_TYPE
        if visits:
            p.tensors = [self.all_visits().to_bytes()]
        else:
            p.tensors = [self.all_infected_visits(day).to_bytes()]

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

