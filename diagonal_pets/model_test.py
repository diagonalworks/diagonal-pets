#!/usr/bin/env python3

if __name__ == "__main__":
    import sys
    sys.path.append("..")

import argparse
import unittest
import numpy as np

import crypto
import model

class ModelTest(unittest.TestCase):

    def __init__(self, name, h, ctxt):
        unittest.TestCase.__init__(self, name)
        self.h = h
        self.ctxt = ctxt

    def setUp(self):
        crypto.init_pyfhel(self.h)

    def test_sample_events(self):
        start_day, stop_day = 10, 56
        infections = np.zeros(1000, dtype=np.uint64)
        selected_people = np.full(len(infections), True, dtype=np.bool_)
        chosen = np.random.choice(np.arange(0, len(infections)), 20, replace=False)
        expected = [pid for (i, pid) in enumerate(chosen) if i % 2 == 0]
        days = {}
        for pid in expected:
            days[pid] = np.random.randint(10, 56)
            infections[pid] = 1 << days[pid]
        unexpected = [pid for (i, pid) in enumerate(chosen) if i % 2 != 0]
        for pid in unexpected:
            infections[pid] = np.uint64(0xffffffffffffffff)
        events = list(model.sample_events(infections, selected_people, start_day, stop_day))
        self.assertEqual(len([1 for (_, _, infected) in events if infected]), len(expected))
        self.assertEqual(len([1 for (_, _, infected) in events if not infected]), len(expected))
        for pid in expected:
            self.assertTrue((days[pid], pid, True) in events)

    def test_sample_events_filters_people(self):
        start_day, stop_day = 10, 56
        infections = np.zeros(1000, dtype=np.uint64)
        selected_people = np.full(len(infections), False, dtype=np.bool_)
        infection_day = 36
        for pid in (10, 20):
            infections[pid] = np.uint64(1 << infection_day)
        for pid in (20, 30):
            selected_people[pid] = True
        events = list(model.sample_events(infections, selected_people, start_day, stop_day))
        self.assertEqual([(infection_day, 20, True), (infection_day, 30, False)], events)

    def test_prepare(self):
        start_day, stop_day, example_day = 0, 56, 10
        people, places = 1000, 100
        infected, person_activities = model.generate_fake_data(people, places, start_day, stop_day)
        pid = model.most_active_person(person_activities)
        visits, infected_visits = model.count_visits(infected, person_activities, start_day, stop_day)
        aggregator = crypto.Aggregator(self.h, self.ctxt)
        model.aggregate(visits, infected_visits, aggregator, self.h)
        visits, infected_visits = model.example(pid, example_day, person_activities, aggregator, self.h, self.ctxt)
        batch = np.zeros((1, model.WINDOW * model.BUCKETS * model.SAMPLES, 1), dtype=np.uint16)
        model.prepare(visits, infected_visits, batch[0])
        # Sanity test the prepared features - they shouldn't be empty,
        # each group should be ordered by the most recent infection count,
        # and infections should increase over time, due to the distribution of
        # out fake data.
        self.assertFalse((batch[0] == 0).all())
        tensor = batch[0].reshape((model.BUCKETS, model.WINDOW, model.SAMPLES))
        for b in range(0, model.BUCKETS):
            for s in range(1, model.SAMPLES):
                self.assertGreaterEqual(tensor[b][-1][s-1], tensor[b][-1][s-1])
        for b in range(0, model.BUCKETS):
            for s in range(0, model.SAMPLES):
                for w in range(1, model.WINDOW):
                    self.assertGreaterEqual(tensor[b][w][s], tensor[b][w-1][s])

    def test_count_infected_visits(self):
        start_day, stop_day = 0, 56
        infected, person_activities = model.generate_fake_data(1000, 100, start_day, stop_day)
        selected_people = np.full(len(person_activities), True, dtype=np.bool_)
        _, infected_visits = model.count_visits(infected, person_activities, start_day, stop_day)
        events = model.sample_events(infected, selected_people, start_day, stop_day)
        for (day, pid, infected_today) in events:
            if infected_today and day > 0:
                for lid in person_activities[pid]:
                    if lid == 0:
                        break
                    self.assertGreater(infected_visits[day-start_day][lid-1], infected_visits[day-start_day-1][lid-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyfhel", action="store_true", help="Use the real Pyfhel library, rather than a fake")
    args = parser.parse_args()
    h, ctxt = crypto.make_pyfhel(fake=not args.pyfhel)
    suite = unittest.TestSuite()
    for method in dir(ModelTest):
        if method.startswith("test_"):
            suite.addTest(ModelTest(method, h, ctxt))
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    main()
