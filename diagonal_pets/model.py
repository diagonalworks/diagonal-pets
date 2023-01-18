import csv
import math
import numpy as np
import tensorflow as tf

from diagonal_pets import crypto
from diagonal_pets import progress

FIT_START_DAY = 0
FIT_STOP_DAY = 56
DAYS = 64
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

def read(data, prefix=""):
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

def aggregate_single_day(day, infected_visits, aggregator, h, track=progress.dont_track):
    aggregator.add_infected_visits(day, aggregator.encrypt_counts(infected_visits[day]))

def sample_events(infected, selected_people, start_day, end_day, track=progress.dont_track):
    if start_day < WINDOW:
        start_day = WINDOW
    if end_day > DAYS:
        end_day = DAYS
    tracker = track("sample_events", len(range(start_day, end_day)))
    for day in range(start_day, end_day):
        masked = np.bitwise_and(infected, np.uint64((1 << day) | (1 << (day - 1))))
        infected_today = np.flatnonzero(np.logical_and(masked == np.uint64((1 << day)), selected_people))
        for pid in infected_today:
            yield (day, pid, True)
        clear = np.flatnonzero(np.logical_and(masked == np.uint64(0), selected_people))
        for pid in np.random.choice(clear, min(len(infected_today), len(clear)), replace=False):
            yield (day, pid, False)
        tracker.next()
    tracker.finish()

# TODO: rename to lookup visits?
def example(pid, day, person_activies, aggregator, h, ctxt):
    places = [lid - 1 for lid in person_activies[pid] if lid != 0]
    visits, infected_visits = aggregator.lookup(places, range(day - WINDOW + 1, day + 1)).decrypt()
    return visits, infected_visits

def examples(events, person_activies, aggregator, h, ctxt, track):
    tracker = track("examples", len(events))
    for (day, pid, infected_today) in events:
        if person_activies[pid][0] != 0:
            visits, infected_visits = example(pid, day, person_activies, aggregator, h, ctxt)
            yield (infected_today, visits, infected_visits)
        tracker.next()
    tracker.finish()

BASE = 6.0
BUCKETS = int(math.log(10000) / math.log(BASE))
SAMPLES = 8

def make_batch(size=1):
    return np.zeros((size, WINDOW * BUCKETS * SAMPLES, 1), dtype=np.uint16)

def prepare(visits, infected_visits, tensor):
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

def prepare_all(examples):
    batch = make_batch()
    target = np.zeros(1, dtype=np.bool_)
    empty = True
    for (infected_today, visits, infected_visits) in examples:
        empty = False
        target[0] = infected_today
        prepare(visits, infected_visits, batch[0])
        yield (batch, target)
    if empty:
        print("prepare_all: empty")

prepared_signature = (tf.TensorSpec(shape=(1, WINDOW * BUCKETS * SAMPLES, 1), dtype=tf.uint16), tf.TensorSpec(shape=(1,), dtype=tf.bool))

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

def predict(h, ctxt, data, track=progress.dont_track):
    infected, person_activities = read(data)
    visits, infected_visits = count_visits(infected, person_activities, FIT_START_DAY, FIT_STOP_DAY, track)
    aggregator = crypto.make_aggregator(h, ctxt)
    aggregator.read(data.aggregator_directory())
    model = tf.keras.models.load_model(data.model_filename())
    pids = [pid for (pid,) in data.preds_format()]
    print("pids:", len(pids))
    batch = make_batch(len(pids))
    t = track("prepare", len(pids))
    for (i, pid) in enumerate(pids):
        visits, infected_visits = example(pid, FIT_STOP_DAY - 1, person_activities, aggregator, h, ctxt)
        prepare(visits, infected_visits, batch[i])
        t.next()
    t.finish()
    predictions = model.predict(x=batch)
    with open(data.preds_dest_filename(), "wt") as f:
        w = csv.writer(f)
        w.writerow(("pid", "score"))
        for (pid, prediction) in zip(pids, predictions):
            w.writerow((str(pid), str(prediction[0])))
