# A disease prediction model based solely on the total number of people
# visiting the places visted by a given person, and the number of those
# people infected within a given number of historical days. Visit counts
# are aggregated homomorphically from all clients by a third party
# (represented by crypto.Aggregator). The third party is unable to see
# the counts associated with each place. Clients decrypt only the place
# counts necessary for each training example or prediction.

import concurrent.futures
import csv
import math
import numpy as np
import tensorflow as tf
import os

from diagonal_pets import crypto
from diagonal_pets import progress

FIT_START_DAY = 0
FIT_STOP_DAY = 56
DAYS = 64

#  The number of days of visit count history to use as input to the model.
WINDOW = 7

def generate_fake_data(people, places, start_day, stop_day):
    infected = np.zeros(people, dtype=np.uint64)
    # exponential growth of infections from start_day to stop_day
    infection_stop = 1
    for day in range(start_day, stop_day):
        for pid in range(0, infection_stop):
            infected[pid] |= np.uint64(1 << day)
        infection_stop = min(infection_stop * 2, people)
    # rescale a zipf distribution to between 4 and people / 2, and
    # use that to generate visits to places
    min_visits, max_visits = min(4, people), int(people / 2)
    visits = np.random.zipf(2.0, places).astype(np.float64)
    visits *= (max_visits - min_visits) / (visits.max() - 1)
    visits += min_visits
    activities = np.zeros(people, np.uint32)
    person_activities = np.zeros((people, int(visits.max())), np.uint32)
    for lid, v in enumerate(visits):
        for pid in np.random.choice(np.arange(0, people), int(v), replace=False):
            person_activities[pid][activities[pid]] = lid + 1
            activities[pid] += 1
    return (infected, person_activities)

def most_active_person(person_activities):
    n = 0
    active_pid = 0
    for pid, activities in enumerate(person_activities):
        nn = count_until_zero(activities)
        if nn > n:
            n = nn
            active_pid = pid
    return active_pid

def count_until_zero(xs):
    n = 0
    for x in xs:
        if x == 0:
            break
        n += 1
    return n

def count_visits(infected, person_activities, start_day, stop_day, track=progress.dont_track):
    visits = np.zeros(len(person_activities), dtype=np.uint32)
    infected_visits = np.zeros((stop_day - start_day, len(person_activities)), np.uint16)
    tracker = track("count_visits", len(person_activities))
    for (pid, places) in enumerate(person_activities):
        tracker.next()
        for lid in places:
            if lid == 0:
                break
            visits[lid-1] += 1
            for day in range(start_day, stop_day):
                if infected[pid] & np.uint64(1 << day):
                    infected_visits[day - start_day][lid-1] += 1
    tracker.finish()
    return (visits, infected_visits)

STATE_INFECTED = "I"

def read(data, prefix="", read_infections=True):
    if prefix != "":
        prefix += " "
    max_alid = 0
    with data.activity_locations() as rows:
        for (alid,) in rows:
            if alid > max_alid:
                max_alid = alid
    print(prefix+"alid:", max_alid)

    max_pid = 0
    with data.person() as rows:
        for (pid,) in rows:
            if pid > max_pid:
                max_pid = pid
    print(prefix+"pid:", max_pid)

    activities = np.zeros(max_pid + 1, np.uint8)
    with data.activity_location_assignment() as rows:
        for (pid, alid) in rows:
            activities[pid] += 1
    max_activities = activities.max()
    print(prefix+"activities:", max_activities)

    infected = np.zeros(max_pid + 1, np.uint64)
    infections = 0
    if read_infections:
        with data.disease_outcome() as rows:
            for (day, pid, state) in rows:
                if state == STATE_INFECTED:
                    infected[pid] |= np.uint64(1 << day)
                    infections += 1
        print(prefix+"infections:", infections)

    activities.fill(0)
    person_activities = np.zeros((max_pid + 1, max_activities), np.uint32)
    visits = np.zeros(max_alid + 1, np.uint16)
    infected_visits = np.zeros((DAYS, max_alid + 1), np.uint16)
    assigned_activities = 0
    with data.activity_location_assignment() as rows:
        for (pid, lid) in rows:
            if lid < max_alid: # Skip activities that aren't at a public place
                person_activities[pid][activities[pid]] = lid + 1
                activities[pid] += 1
                visits[lid] += 1
                for day in range(0, DAYS):
                    if infected[pid] & np.uint64(1 << day):
                        infected_visits[day][lid] += 1
                assigned_activities += 1
    print(prefix+"assigned_activities:", assigned_activities)
    return (infected, person_activities)

def aggregate(visits, infected_visits, aggregator, h, track=progress.dont_track):
    aggregator.add_visits(aggregator.encrypt_counts(visits))
    tracker = track("aggregate", len(infected_visits))
    for day in range(0, len(infected_visits)):
        tracker.next()
        aggregator.add_infected_visits(day, aggregator.encrypt_counts(infected_visits[day]))
    tracker.finish()

# aggregate visits made, writing completed days to disk and forgetting them to
# save memory.
def aggregate_and_write(visits, infected_visits, aggregator, directory, track=progress.dont_track):
    aggregator.add_visits(aggregator.encrypt_counts(visits))
    aggregator.write(directory, visits=True)
    t = track("aggregate", len(infected_visits))
    def _aggregate(day):
        aggregator.add_infected_visits(day, aggregator.encrypt_counts(infected_visits[day]))
        aggregator.write(directory, day=day)
        aggregator.clear_infected_visits(day, day+1)
        t.next()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as e:
        fs = [e.submit(_aggregate, day) for day in range(0, len(infected_visits))]
        for f in concurrent.futures.as_completed(fs):
            e = f.exception()
            if e is not None:
                raise e
    t.finish()

def aggregate_single_day(day, infected_visits, aggregator, h, track=progress.dont_track):
    aggregator.add_infected_visits(day, aggregator.encrypt_counts(infected_visits[day]))

def newly_positive_events(infected, start_day, end_day):
    for day in range(max(start_day, 1), end_day):
        mask = np.uint64(1 << day) | np.uint64(1 << (day - 1))
        newly_positive = np.flatnonzero(np.bitwise_and(infected, mask) == np.uint64(1 << day))
        for pid in newly_positive:
            yield (day, pid)

# The number of days into the future to consider an infection leading to a
# positive training example. In the ideal case, this would be 7, as we're
# attempting to predict an infection within a week. We only use features
# associated with places, however, so raising this value amplifies noise.
# We emperially set a value of 3 based on the results of test fit runs.
HORIZON = 3

# Return an iterator of (day, pid, infected within horizon) to use as the basis
# of training examples. We attempt to balance the number of positive and negative
# events generated from each day.
def sample_events(infected, selected_people, start_day, end_day, track=progress.dont_track):
    if start_day < WINDOW:
        start_day = WINDOW
    if end_day > DAYS:
        end_day = DAYS
    tracker = track("sample_events", len(range(start_day, end_day)))
    for day in range(start_day, end_day):
        clear_yesterday = np.bitwise_and(infected,  np.uint64(1 << (day - 1))) == np.uint64(0)
        infection_mask = np.uint64(0)
        for infection_day in range(day, min(day + HORIZON, DAYS)):
            infection_mask |= np.uint64(1 << infection_day)
        infected_within_horizon = np.bitwise_and(infected, infection_mask) != np.uint64(0)
        infected_pids = np.flatnonzero(np.logical_and(np.logical_and(clear_yesterday, infected_within_horizon), selected_people))
        for pid in infected_pids:
            yield (day, pid, True)
        clear_mask = infection_mask |  np.uint64(1 << (day - 1))
        clear = np.flatnonzero(np.logical_and(np.bitwise_and(infected, clear_mask) == np.uint64(0), selected_people))
        for pid in np.random.choice(clear, min(len(infected_pids), len(clear)), replace=False):
            yield (day, pid, False)
        tracker.next()
    tracker.finish()

# Use aggregator to lookup the visit and infected visit counts associated with
# each place visited by pid on this day and the previous WINDOW - 1.
def lookup_visits(pid, day, person_activies, aggregator):
    places = [lid - 1 for lid in person_activies[pid] if lid != 0]
    visits, infected_visits = aggregator.lookup(places, range(day - WINDOW + 1, day + 1)).decrypt()
    return visits, infected_visits

def lookup_all_visits(events, person_activies, aggregator, directory):
    last_day = WINDOW
    aggregator.reference_window(directory, last_day - WINDOW + 1, last_day + 1)
    for (day, pid, infected_today) in events:
        if person_activies[pid][0] != 0:
            if day != last_day:
                aggregator.swap_window(directory, last_day - WINDOW + 1, last_day + 1, day - WINDOW + 1, day + 1)
                last_day = day
            visits, infected_visits = lookup_visits(pid, day, person_activies, aggregator)
            yield (infected_today, visits, infected_visits)
    aggregator.unreference_window(last_day - WINDOW + 1, last_day + 1)

# The input of our model is the total number of people visiting places
# visted by a given individual, and the number of people visiting infected
# on the current day, and the previous VISIT - 1. We divide these counts
# into groups of SAMPLES, bucketed by the log6() of their total visit count.
# Places are ordered within each bucket by the number of infected people
# visiting on the most recent day.
# log6 is chosen as 6^x for x 0..log6(10000) gives a series of buckets that
# naturally reflect a range of different types of venue visited by an
# individual.
#
# We feed these values into 1D convolutional layers for each bucket,
# before aggregating via two dense layers.

BASE = 6.0
BUCKETS = int(math.log(10000) / math.log(BASE))
SAMPLES = 8

def make_model():
    inputs = tf.keras.Input(shape=(BUCKETS * WINDOW * SAMPLES, 1))
    buckets = []
    for i in range(0, BUCKETS):
        n = WINDOW * SAMPLES
        infections = tf.keras.layers.Cropping1D(cropping=(i * n, (BUCKETS - i - 1) * n))(inputs)
        conved = tf.keras.layers.Conv1D(SAMPLES, input_shape=(WINDOW * SAMPLES, 1), kernel_size=WINDOW, strides=(SAMPLES,), activation="relu")(infections)
        reshaped = tf.keras.layers.Reshape((WINDOW * SAMPLES,))(conved)
        buckets.append(reshaped)
    merged = tf.keras.layers.Concatenate(axis=1)(buckets)
    dense1 = tf.keras.layers.Dense(BUCKETS * SAMPLES, activation="relu")(merged)
    dense2 = tf.keras.layers.Dense(BUCKETS * SAMPLES, activation="relu")(dense1)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(dense2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def make_batch(size=1):
    return np.zeros((size, WINDOW * BUCKETS * SAMPLES, 1), dtype=np.uint16)

def to_model_input(visits, infected_visits, tensor):
    buckets = [[] for i in range(0, BUCKETS)]
    for (i, count) in enumerate(visits):
        if count > 0:
            b = int(math.log(float(count)) / math.log(BASE))
            if b >= len(buckets):
                b = len(buckets) - 1
            buckets[b].append(i)
    for (i, b) in enumerate(buckets):
        b.sort(key=lambda j: -int(infected_visits[-1][j]))
        while len(b) < SAMPLES:
            b.append(-1)
        buckets[i] = b[0:SAMPLES]
    i = 0
    for b in buckets:
        for w in range(0, WINDOW):
            for j in b:
                if j >= 0:
                    tensor[i][0] = infected_visits[w][j]
                else:
                    tensor[i][0] = 0.0
                i += 1

def to_model_input_all(features, t):
    i = 0
    batch = make_batch()
    target = np.zeros(1, dtype=np.bool_)
    for (infected_today, visits, infected_visits) in features:
        target[0] = infected_today
        to_model_input(visits, infected_visits, batch[0])
        yield (batch, target)
        if i % 100 == 0:
            t.next()
        i += 1
    t.finish()

input_signature = (tf.TensorSpec(shape=(1, WINDOW * BUCKETS * SAMPLES, 1), dtype=tf.uint16), tf.TensorSpec(shape=(1,), dtype=tf.bool))

def fit(infected, person_activities, model, aggregator, data, track, epochs=2):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    selected_people = np.add.reduce(person_activities, axis=1) > 0
    events = list(sample_events(infected, selected_people, FIT_START_DAY, FIT_STOP_DAY, track))
    def prepared():
        t = track("fit", len(events)/100)
        features = lookup_all_visits(events, person_activities, aggregator, data.aggregator_directory())
        return to_model_input_all(features, t)
    input = tf.data.Dataset.from_generator(prepared, output_signature=input_signature)
    model.fit(x=input, epochs=epochs, verbose=2)
    return len(events)

def predict(h, ctxt, data, track=progress.dont_track):
    _, person_activities = read(data, read_infections=False)
    aggregator = crypto.make_aggregator(h, ctxt)
    aggregator.read(data.aggregator_directory(), visits=True)
    aggregator.read_infected_visits_window(data.aggregator_directory(), FIT_STOP_DAY - WINDOW, FIT_STOP_DAY)
    model = tf.keras.models.load_model(data.model_filename())
    pids = [pid for (pid,) in data.preds_format()]
    print("pids:", len(pids))
    batch = make_batch(len(pids))
    t = track("prepare", len(pids)/1000)
    for i, pid in enumerate(pids):
        visits, infected_visits = lookup_visits(pid, FIT_STOP_DAY - 1, person_activities, aggregator)
        to_model_input(visits, infected_visits, batch[i])
        if i % 1000 == 0:
            t.next()
    t.finish()
    predictions = model.predict(x=batch, workers=os.cpu_count(), verbose=2)
    with open(data.preds_dest_filename(), "wt") as f:
        w = csv.writer(f)
        w.writerow(("pid", "score"))
        for (pid, prediction) in zip(pids, predictions):
            w.writerow((str(pid), str(prediction[0])))
