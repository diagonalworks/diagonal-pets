#!/usr/bin/env python3
# Generate infection data for testing.

import argparse
import csv
import gzip
import random
import numpy as np

import diagonal_pets.features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="dw", help="The prefix for test data filenames")
    parser.add_argument("--people", default=1000, help="The number of people to represent")
    parser.add_argument("--days", default=56, help="The number of days to represent")
    parser.add_argument("--places", default=100, help="The number of places to represent")
    args = parser.parse_args()

    start_day, stop_day = 0, args.days
    infected, person_activities = diagonal_pets.features.generate_fake_data(args.people, args.places, start_day, stop_day)

    with gzip.open("%s_person.csv.gz" % args.prefix, "wt", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(("pid","age"))
        for i in range(0, args.people):
            w.writerow((str(i), str(random.randint(0, 100))))

    with gzip.open("%s_activity_locations.csv.gz" % args.prefix, "wt", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(("alid",))
        for i in range(0, args.places):
            w.writerow((str(i),))

    with gzip.open("%s_activity_location_assignment.csv.gz" % args.prefix, "wt", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(("pid", "lid"))
        for pid in range(0, args.people):
            for lid in person_activities[pid]:
                if lid == 0:
                    break
                w.writerow((str(pid), str(lid-1)))

    with gzip.open("%s_disease_outcome_training.csv.gz" % args.prefix, "wt", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(("day", "pid", "state"))
        for day in range(0, args.days):
            for pid in range(0, args.people):
                mask = np.uint64(1 << day)
                state = (infected[pid] & mask) == mask
                w.writerow((str(day), str(pid), ["S", "I"][int(state)]))

    with gzip.open("%s_disease_outcome_target.csv.gz" % args.prefix, "wt", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(("pid",))
        for pid in range(0, args.people):
            w.writerow((str(pid),))

if __name__ == "__main__":
    main()