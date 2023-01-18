#!/usr/bin/env python3

import sys
sys.path.append("/code_execution/src/")

FAKE_PYFHEL=True

import argparse
import csv
import random
import tensorflow as tf
import numpy as np

from pathlib import Path

import diagonal_pets

FIT_START_DAY = 0
FIT_STOP_DAY = 56
DAYS = 64

def paths_from_flags(flags):
    input = Path(flags.input)
    return {
        "person_data_path": input / ("%s_person.csv.gz" % flags.prefix),
        "activity_location_data_path":  input / ("%s_activity_locations.csv.gz" % flags.prefix),
        "activity_location_assignment_data_path": input / ("%s_activity_location_assignment.csv.gz" % flags.prefix),
        "disease_outcome_data_path": input / ("%s_disease_outcome_training.csv.gz" % flags.prefix),
        "model_dir": Path(flags.output),
        "preds_format_path": Path(flags.output) / ("%s_disease_outcome_target.csv.gz" % flags.prefix),
        "preds_dest_path": Path(flags.output) / ("%s_predictions.csv" % flags.prefix),
        "prefix": flags.prefix,
    }

def fit(fake_pyfhel=FAKE_PYFHEL, **args):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    diagonal_pets.init_pyfhel(h)
    h.keyGen()
    h.rotateKeyGen()
    diagonal_pets.init_pyfhel(h)
    data = diagonal_pets.make_file_data(**args)
    infected, person_activities = diagonal_pets.read(data)
    visits, infected_visits = diagonal_pets.count_visits(infected, person_activities, FIT_START_DAY, FIT_STOP_DAY, diagonal_pets.track)
    aggregator = diagonal_pets.make_aggregator(h, ctxt)
    diagonal_pets.aggregate(visits, infected_visits, aggregator, h, diagonal_pets.track)
    selected_people = np.add.reduce(person_activities, axis=1) > 0
    model = diagonal_pets.make_model()
    events = list(diagonal_pets.sample_events(infected, selected_people, FIT_START_DAY, FIT_STOP_DAY, diagonal_pets.track))
    def prepared():
        examples = diagonal_pets.examples(events, person_activities, aggregator, h, ctxt, diagonal_pets.dont_track)
        return diagonal_pets.prepare_all(examples)
    input = tf.data.Dataset.from_generator(prepared, output_signature=diagonal_pets.prepared_signature)
    model.fit(x=input, epochs=4)
    model.save(data.model_filename())
    aggregator.clear_infected_visits(0, FIT_STOP_DAY - diagonal_pets.WINDOW)
    aggregator.write(data.aggregator_directory())
    diagonal_pets.write_keys(data, h)

def predict(fake_pyfhel=FAKE_PYFHEL, **args):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    diagonal_pets.init_pyfhel(h)
    data = diagonal_pets.make_file_data(**args)
    diagonal_pets.read_keys(data, h)
    return diagonal_pets.predict(h, ctxt, data, diagonal_pets.track)

def score(score_prefix, **args):
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, roc_curve, RocCurveDisplay

    data = diagonal_pets.make_file_data(**args)
    target = dict(data.targets())
    predictions = list(data.preds())
    y_true = [int(target.get(pid, 0)) for (pid, score) in predictions]
    y_score = [score for (pid, score) in predictions]
    print("ap: ", average_precision_score(y_true, y_score))
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="Fit a model")
    parser.add_argument("--predict", action="store_true", help="Generate predictions using a model")
    parser.add_argument("--score", action="store_true", help="Score generated predictions against the target")
    parser.add_argument("--prefix", default="dw", help="The prefix for input data filenames")
    parser.add_argument("--input", default=".", help="The directory containing training data")
    parser.add_argument("--output", default=".", help="The directory in which to write the model")
    parser.add_argument("--pyfhel", action="store_true", help="Use the real Pyfhel library, rather than a fake")
    parser.add_argument("--preds_prefix", default=None, help="Prefix of output to score, if not --prefix")
    flags = parser.parse_args()
    if flags.fit:
        fit(fake_pyfhel=not flags.pyfhel, **paths_from_flags(flags))
    if flags.predict:
        predict(fake_pyfhel=not flags.pyfhel, **paths_from_flags(flags))
    if flags.score:
        score(score_prefix=flags.preds_prefix, **paths_from_flags(flags))

if __name__ == "__main__":
    main()