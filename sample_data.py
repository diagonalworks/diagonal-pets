#!/usr/bin/env python3
# Write a new set of training containing a given percentage of the
# individuals from the original data set.
# Filenames will include the sampled percentage, for example, a
# 1% sample of the va dataset will have files following the
# pattern va1_activity_locations.csv.gz

import argparse
import csv
import gzip

from pathlib import Path

FILENAMES = [
    "%s_person.csv.gz",
    "%s_activity_locations.csv.gz",
    "%s_activity_location_assignment.csv.gz",
    "%s_disease_outcome_training.csv.gz",
    "%s_disease_outcome_target.csv.gz",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="va", help="The prefix for input data filenames")
    parser.add_argument("--input", default=".", help="The directory containing input data")
    parser.add_argument("--percentage", default=1, type=int, help="Percentage of individuals to sample")
    flags = parser.parse_args()

    for filename in FILENAMES:
        p = Path(flags.input).joinpath(Path(filename % flags.prefix))
        with gzip.open(p, "rt") as input:
            with gzip.open(filename % (flags.prefix + str(flags.percentage)), "wt") as output:
                r = csv.reader(input)
                w = csv.writer(output)
                headers = next(r)
                if "pid" in headers:
                    pid = headers.index("pid")
                    w.writerow(headers)
                    for row in r:
                        if int(row[pid]) % 100 <= flags.percentage:
                            w.writerow(row)
                else:
                    w.writerow(headers)
                    for row in r:
                        w.writerow(row)

if __name__ == "__main__":
    main()