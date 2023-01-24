#!/usr/bin/env python3
# Write a new set of training containing a given percentage of the
# individuals from the original data set.
# Filenames will include the sampled percentage, for example, a
# 1% sample of the va dataset will have files following the
# pattern va1_activity_locations.csv.gz

import argparse
import csv
import random
import gzip

from pathlib import Path

FILENAMES = [
    "%s_person%s.csv.gz",
    "%s_activity_locations%s.csv.gz",
    "%s_activity_location_assignment%s.csv.gz",
    "%s_disease_outcome_training%s.csv.gz",
    "%s_disease_outcome_target%s.csv.gz",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="va", help="The prefix for input data filenames")
    parser.add_argument("--input", default=".", help="The directory containing input data")
    parser.add_argument("--percentage", default=1, type=int, help="Percentage of individuals to sample")
    parser.add_argument("--shards", default=1, type=int, help="Split the output into the given number of shards, by pid")
    flags = parser.parse_args()

    for filename in FILENAMES:
        p = Path(flags.input).joinpath(Path(filename % (flags.prefix, "")))
        with gzip.open(p, "rt") as input:
            if flags.shards == 1:
                outputs = [gzip.open(filename % (flags.prefix + str(flags.percentage), ""), "wt")]
            else:
                outputs = [gzip.open(filename % (flags.prefix + str(flags.percentage), ".%d" % i), "wt") for i in range(0, flags.shards)]
            r = csv.reader(input)
            ws = [csv.writer(o) for o in outputs]
            headers = next(r)
            if "pid" in headers:
                pid = headers.index("pid")
                for w in ws:
                    w.writerow(headers)
                for row in r:
                    if int(row[pid]) % 100 <= flags.percentage:
                        ws[hash(row[pid]) % flags.shards].writerow(row)
            else:
                for w in ws:
                    w.writerow(headers)
                for row in r:
                    for w in ws:
                        w.writerow(row)
            for o in outputs:
                o.close()

if __name__ == "__main__":
    main()