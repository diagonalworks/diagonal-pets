import io
import numpy as np
import random
import struct

GROUP_SIZE = 32
DTYPE = np.int64

def make_pyfhel(fake=False):
    if fake:
        return FakePyfhel(), FakePyCtxt
    else:
        import Pyfhel
        return Pyfhel.Pyfhel(), Pyfhel.PyCtxt

def init_pyfhel(h):
    h.contextGen(scheme="BFV", n=2*GROUP_SIZE, t_bits=20, sec=128)
    h.keyGen()
    h.rotateKeyGen()

class GroupedCounts:

    def __init__(self, h, ctxt):
        self.h = h
        self.ctxt = ctxt
        self.counts = None

    def add(self, counts):
        if self.counts is None:
            self.counts = [self.ctxt(bytestring=b) for b in counts]
        else:
            for (c1, c2) in zip(self.counts, counts):                
                self.h.add(c1, self.ctxt(bytestring=c2))

    def lookup(self, indicies):
        if len(indicies) == 0 or self.counts is None:
            return []
        seen = [False] * (int(max(indicies) / GROUP_SIZE) + 1)
        values = []
        for i in range(0, len(indicies)):
            group = int(indicies[i] / GROUP_SIZE)
            if seen[group] or group >= len(self.counts):
                continue
            else:
                seen[group] = True
            g = self.counts[group].copy()
            self.h.rotate(g, indicies[i] % GROUP_SIZE)
            mask = np.zeros(self.h.n, dtype=DTYPE)
            mask[0] = 1
            self.h.multiply_plain(g, mask)
            values.append(g.to_bytes())
        return values

class Aggregator:

    DAYS = 64

    def __init__(self, h, ctxt):
        self.visits = GroupedCounts(h, ctxt)
        self.infected_visits = [GroupedCounts(h, ctxt) for i in range(0, self.DAYS)]

    def add_visits(self, counts):
        self.visits.add(counts)

    def add_infected_visits(self, day, counts):
        self.infected_visits[day].add(counts)

    def lookup(self, indicies, days):
        visits = self.visits.lookup(indicies)
        infected_visits = [self.infected_visits[day].lookup(indicies) for day in days]
        shuffle = list(range(0, len(visits)))
        random.shuffle(shuffle)
        visits = [visits[i] for i in shuffle]
        infected_visits = [[infected_visits[day][i] for i in shuffle] for day in range(0, len(days))]
        return (visits, infected_visits)

def make_aggregator(h, ctxt):
    return Aggregator(h, ctxt)

def encrypt_counts(counts, h):
    serialised = []
    for i in range(0, len(counts), GROUP_SIZE):
        group = np.zeros(GROUP_SIZE, dtype=DTYPE)
        for j in range(0, min(GROUP_SIZE, len(counts) - i)):
            group[j] = counts[i+j]
        serialised.append(h.encryptPtxt(h.encodeInt(group)).to_bytes())
    return serialised

def decrypt_lookup(lookup, h, ctxt):
    visits, infected_visits = lookup
    return (
        [h.decryptInt(ctxt(bytestring=b))[0] for b in visits],
        [[h.decryptInt(ctxt(bytestring=b))[0] for b in day] for day in infected_visits])


# A fake implementation of a subset of the homomorphic crypto
# routines from Pyfhel, for testing in environments where
# Pyfhel isn't present. Operations are performed in the clear.
class FakePyfhel:

    def contextGen(self, **args):
        self.n = args["n"]

    def keyGen(self):
        pass

    def rotateKeyGen(self):
        pass

    def encodeInt(self, v):
        if len(v) > self.n:
            raise ValueError("Length %d is beyond maximum of %d" % (len(v), self.n))    
        return FakePyPtxt(np.pad(v, (0, self.n - len(v))))

    def encryptPtxt(self, p):
        if not isinstance(p, FakePyPtxt):
            raise TypeError("Expected FakePyPtxt")
        return FakePyCtxt(p.v)

    def decryptInt(self, c):
        if not isinstance(c, FakePyCtxt):
            raise TypeError("Expected PyCtxt")
        return c.v

    def add(self, c1, c2):
        if not isinstance(c1, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        if not isinstance(c2, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        c1.v += c2.v

    def add_plain(self, c, v):
        if not isinstance(c, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        c.v += v

    def multiply_plain(self, c, v):
        if not isinstance(c, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")    
        c.v *= v

    def rotate(self, c, steps):
        if not isinstance(c, FakePyCtxt):
            raise TypeError("Expected FakePyCtxt")
        if abs(steps) >= self.n / 2:
            raise ValueError("Rotate steps %d beyond maximum of %d" % (steps, self.n/2-1))
        c.v = np.roll(c.v, -steps)

class FakePyPtxt:

    def __init__(self, v):
        self.v = v

class FakePyCtxt:

    MAGIC = b"\xe4\xd4\xd8\xd2"

    def __init__(self, v=None, bytestring=None):
        if v is not None:
            self.v = v
        elif bytestring is not None:
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

