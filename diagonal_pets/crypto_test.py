#!/usr/bin/env python3

import argparse
import io
import numpy as np
import tempfile
import unittest
from pathlib import Path

import crypto

class AggregatorTest(unittest.TestCase):

    def __init__(self, name, h, ctxt):
        unittest.TestCase.__init__(self, name)
        self.h = h
        self.ctxt = ctxt

    def setUp(self):
        crypto.init_pyfhel(self.h)
        self.h.keyGen()
        self.h.rotateKeyGen()

    def test_aggregate(self):
        places = 1321
        sample = [832, 513, 961, 315]
        sampled_days = list(range(2, 8))
        aggregator = crypto.Aggregator(self.h, self.ctxt)
        expected_visits = np.random.randint(0, 1000, (places,), crypto.DTYPE)
        aggregator.add_visits(aggregator.encrypt_counts(expected_visits))
        expected_infected_visits = [np.zeros(places, crypto.DTYPE) for day in sampled_days]
        for day in range(0, 10):
            for shard in range(0, 5):
                counts = np.random.randint(0, 5, (places,), crypto.DTYPE)
                aggregator.add_infected_visits(day, aggregator.encrypt_counts(counts))
                if day in sampled_days:
                    i = sampled_days.index(day)
                    expected_infected_visits[i] += counts
        visits, infections = aggregator.lookup(sample, sampled_days).decrypt()
        for s in sample:
            if expected_visits[s] not in visits:
                self.fail("Failed to find expected visits")
        for (i, day) in enumerate(sampled_days):
            for s in sample:
                if expected_infected_visits[i][s] not in infections[i]:
                    self.fail("Failed to find expected infections for day %d" % day)

    def test_clear(self):
        aggregator = crypto.Aggregator(self.h, self.ctxt)
        places = 1321
        cleared_visits = np.random.randint(0, 1000, (places,), crypto.DTYPE)
        expected_visits = np.random.randint(0, 1000, (places,), crypto.DTYPE)
        aggregator.add_infected_visits(36, aggregator.encrypt_counts(cleared_visits))
        aggregator.clear_infected_visits(30, 40)
        aggregator.add_infected_visits(36, aggregator.encrypt_counts(expected_visits))
        self.assertTrue((expected_visits == aggregator.all_infected_visits(36).decrypt()[0:places]).all())

    def test_file_round_trip(self):
        places = 1321
        visits = np.random.randint(0, 1000, (places,), crypto.DTYPE)
        infected_visits = np.random.randint(0, 1000, (places,), crypto.DTYPE)
        aggregator = crypto.Aggregator(self.h, self.ctxt)
        aggregator.add_visits(aggregator.encrypt_counts(visits))
        aggregator.add_infected_visits(36, aggregator.encrypt_counts(infected_visits))
        restored_aggregator = crypto.Aggregator(self.h, self.ctxt)
        with tempfile.TemporaryDirectory() as d:
            aggregator.write(Path(d))
            restored_aggregator.read(Path(d))
        counts = restored_aggregator.all_visits().decrypt()
        self.assertTrue((counts[0:len(visits)] == visits).all())
        counts = restored_aggregator.all_infected_visits(36).decrypt()
        self.assertTrue((counts[0:len(infected_visits)] == infected_visits).all())

class FakePyfhelTest(unittest.TestCase):

    def setUp(self):
        self.h = crypto.FakePyfhel()
        self.h.contextGen(scheme="BFV", n=2**4)
        self.h.keyGen()
        self.h.rotateKeyGen()

    def test_add(self):
        p1 = self.h.encodeInt(np.arange(0, self.h.n, dtype=np.int64))
        c1 = self.h.encryptPtxt(p1)
        p2 = self.h.encodeInt(np.full(self.h.n, 1))
        c2 = self.h.encryptPtxt(p2)
        self.h.add(c1, c2)
        self.assertTrue(np.array_equal(self.h.decryptInt(c1), np.arange(1, self.h.n+1, dtype=np.int64)))

    def test_add_plain(self):
        p = self.h.encodeInt(np.arange(0, self.h.n, dtype=np.int64))
        c = self.h.encryptPtxt(p)
        self.h.add_plain(c, self.h.encodeInt(np.full(self.h.n, 1)))
        self.assertTrue(np.array_equal(self.h.decryptInt(c), np.arange(1, self.h.n+1, dtype=np.int64)))

    def test_multiply_plain(self):
        v = np.arange(0, self.h.n, dtype=np.int64)
        p = self.h.encodeInt(v)
        c = self.h.encryptPtxt(p)
        self.h.multiply_plain(c, self.h.encodeInt(np.full(len(v), 2)))
        self.assertTrue(np.array_equal(self.h.decryptInt(c), v * 2))

    def test_rotate(self):
        p = self.h.encodeInt(np.arange(0, self.h.n, dtype=np.int64))
        c = self.h.encryptPtxt(p)
        self.h.rotate(c, 1)
        expected = np.arange(1, self.h.n+1, dtype=np.int64)
        expected[len(expected)-1] = 0
        self.assertTrue(np.array_equal(self.h.decryptInt(c), expected))

    def test_to_and_from_bytes(self):
        v = np.arange(0, self.h.n, dtype=np.int64)
        b = self.h.encryptPtxt(self.h.encodeInt(v)).to_bytes()
        vv = crypto.FakePyCtxt(bytestring=b, pyfhel=self.h)
        self.assertTrue(np.array_equal(self.h.decryptInt(vv), v))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyfhel", action="store_true", help="Use the real Pyfhel library, rather than a fake")
    args = parser.parse_args()

    suite = unittest.TestSuite()
    for method in dir(AggregatorTest):
        if method.startswith("test_"):
            h, ctxt = crypto.make_pyfhel(fake=not args.pyfhel)
            suite.addTest(AggregatorTest(method, h, ctxt))
    for method in dir(FakePyfhelTest):
        if method.startswith("test_"):
            suite.addTest(FakePyfhelTest(method))
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    main()