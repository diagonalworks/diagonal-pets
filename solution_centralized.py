#!/usr/bin/env python3

import argparse
import csv
import random
import tensorflow as tf
import numpy as np

from pathlib import Path

import diagonal_pets

START_DAY = 0
STOP_DAY = 56

def paths_from_flags(flags):
    directory = Path(flags.input)
    return {
        "person_data_path": directory.joinpath(Path("%s_person.csv.gz" % flags.prefix)),
        "activity_location_data_path":  directory.joinpath(Path("%s_activity_locations.csv.gz" % flags.prefix)),
        "activity_location_assignment_data_path": directory.joinpath(Path("%s_activity_location_assignment.csv.gz" % flags.prefix)),
        "disease_outcome_data_path": directory.joinpath(Path("%s_disease_outcome_training.csv.gz" % flags.prefix)),
        "model_dir": Path(flags.output),
        "preds_format_path": Path(flags.output).joinpath(Path("%s_disease_outcome_target.csv.gz" % flags.prefix)),
        "preds_dest_path": Path(flags.output).joinpath(Path("%s_predictions.csv" % flags.prefix)),
    }

def fit(fake_pyfhel=False, **args):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    diagonal_pets.init_pyfhel(h)
    data = diagonal_pets.make_file_data(**args)
    infected, person_activities = diagonal_pets.read(data)
    visits, infected_visits = diagonal_pets.count_visits(infected, person_activities, START_DAY, STOP_DAY, diagonal_pets.track)
    aggregator = diagonal_pets.make_aggregator(h, ctxt)
    diagonal_pets.aggregate(visits, infected_visits, aggregator, h, diagonal_pets.track)
    selected_people = np.add.reduce(person_activities, axis=1) > 0
    model = diagonal_pets.make_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    events = list(diagonal_pets.sample_events(infected, selected_people, START_DAY, STOP_DAY, diagonal_pets.track))
    def prepared():
        examples = diagonal_pets.examples(events, person_activities, aggregator, h, ctxt, diagonal_pets.dont_track)
        return diagonal_pets.prepare_all(examples)
    input = tf.data.Dataset.from_generator(prepared, output_signature=diagonal_pets.prepared_signature)
    model.fit(x=input, epochs=4)
    model.save(data.model_filename())

def predict(fake_pyfhel=False, **args):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    diagonal_pets.init_pyfhel(h)
    data = diagonal_pets.make_file_data(**args)
    infected, person_activities = diagonal_pets.read(data)
    visits, infected_visits = diagonal_pets.count_visits(infected, person_activities, START_DAY, STOP_DAY, diagonal_pets.track)
    aggregator = diagonal_pets.make_aggregator(h, ctxt)
    diagonal_pets.aggregate(visits, infected_visits, aggregator, h, diagonal_pets.track)    
    model = tf.keras.models.load_model(data.model_filename())
    pids = [pid for (pid,) in data.preds_format()]
    print("pids:", len(pids))
    batch = diagonal_pets.make_batch(len(pids))
    for (i, pid) in enumerate(pids):
        visits, infected_visits = diagonal_pets.example(pid, STOP_DAY - 1, person_activities, aggregator, h, ctxt)
        diagonal_pets.prepare(visits, infected_visits, batch[i])
    predictions = model.predict(x=batch)
    with open(data.preds_dest_filename(), "wt") as f:
        w = csv.writer(f)
        w.writerow(("pid", "score"))
        for (pid, prediction) in zip(pids, predictions):
            w.writerow((str(pid), str(prediction[0])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="Fit a model")
    parser.add_argument("--predict", action="store_true", help="Generate predictions using a model")
    parser.add_argument("--prefix", default="dw", help="The prefix for input data filenames")
    parser.add_argument("--input", default=".", help="The directory containing training data")
    parser.add_argument("--output", default=".", help="The directory in which to write the model")
    parser.add_argument("--pyfhel", action="store_true", help="Use the real Pyfhel library, rather than a fake")
    flags = parser.parse_args()
    if flags.fit:
        fit(fake_pyfhel=not flags.pyfhel, **paths_from_flags(flags))
    if flags.predict:
        predict(fake_pyfhel=not flags.pyfhel, **paths_from_flags(flags))

if __name__ == "__main__":
    main()