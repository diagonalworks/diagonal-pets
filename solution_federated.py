#!/usr/bin/env python3

import sys
sys.path.append("/code_execution/src/")

FAKE_PYFHEL=True

import argparse
import flwr as fl
import os

from pathlib import Path

import diagonal_pets

def train_setup(server_dir, client_dirs_dict, fake_pyfhel=FAKE_PYFHEL):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    h.keyGen()
    h.rotateKeyGen()
    data = diagonal_pets.make_file_data(server_dir=server_dir)
    diagonal_pets.write_keys(data, h, secret=False, public=True, rotate=True)
    for _, client_dir in client_dirs_dict.items():
        data = diagonal_pets.make_file_data(client_dir=client_dir)
        diagonal_pets.write_keys(data, h, secret=True, public=True, rotate=True)

def train_strategy_factory(server_dir, fake_pyfhel=FAKE_PYFHEL, prefix="dw"):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    data = diagonal_pets.make_file_data(server_dir=server_dir)
    diagonal_pets.read_keys(data, h, secret=False, public=True, rotate=True)
    strategy = fl.server.strategy.FedAvg()
    return diagonal_pets.FitStrategy(h, ctxt, strategy, prefix), diagonal_pets.FIT_ROUNDS

def train_client_factory(cid, fake_pyfhel=FAKE_PYFHEL, client_dir=None, **args):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    data = diagonal_pets.make_file_data(client_dir=client_dir)
    diagonal_pets.read_keys(data, h, secret=True, public=True, rotate=True)
    return diagonal_pets.FitClient(cid, h, ctxt, client_dir=client_dir, **args)

def test_strategy_factory(server_dir, fake_pyfhel=FAKE_PYFHEL, prefix="dw"):
    return diagonal_pets.TestStrategy(), diagonal_pets.TEST_ROUNDS

def test_client_factory(cid, fake_pyfhel=FAKE_PYFHEL, client_dir=None, **args):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    data = diagonal_pets.make_file_data(client_dir=client_dir)
    diagonal_pets.read_keys(data, h, secret=True, public=True, rotate=True)
    return diagonal_pets.TestClient(cid, h, ctxt, client_dir=client_dir, **args)

def paths_from_flags(flags, cid):
    input = Path(flags.input)
    return {
        "person_data_path": input / ("%s_person.%s.csv.gz" % (flags.prefix, cid)),
        "activity_location_data_path":  input / ("%s_activity_locations.%s.csv.gz" % (flags.prefix, cid)),
        "activity_location_assignment_data_path": input / ("%s_activity_location_assignment.%s.csv.gz" % (flags.prefix, cid)),
        "disease_outcome_data_path": input / ("%s_disease_outcome_training.%s.csv.gz" % (flags.prefix, cid)),
        "model_dir": Path(flags.output),
        "preds_format_path": Path(flags.output) / ("%s_disease_outcome_target.%s.csv.gz" % (flags.prefix, cid)),
        "preds_dest_path": Path(flags.output) / ("%s_predictions.%s.csv" % (flags.prefix, cid)),
        "prefix": flags.prefix,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="Fit a model")
    parser.add_argument("--predict", action="store_true", help="Generate predictions using a model")
    parser.add_argument("--prefix", default="dw", help="The prefix for input data filenames")
    parser.add_argument("--input", default=".", help="The directory containing training data")
    parser.add_argument("--output", default=".", help="The directory in which to write the model")
    parser.add_argument("--clients", default=3, type=int, help="Number of clients to start")
    parser.add_argument("--pyfhel", action="store_true", help="Use the real Pyfhel library, rather than a fake")
    parser.add_argument("--fake-pyfhel", dest="fake_pyfhel", action="store_true", help="Use the real Pyfhel library, rather than a fake")
    flags = parser.parse_args()

    fake_pyfhel = FAKE_PYFHEL
    if flags.pyfhel and flags.fake-pyfhel:
        print("Can only specify one of --pyfhel and --fake-pyfhel")
        return
    if flags.pyfhel:
        fake_pyfhel = False
    elif flags.fake_pyfhel:
        fake_pyfhel = True

    client_dirs = dict([(str(i), Path(flags.output) / "client-tmp" / str(i)) for i in range(0, flags.clients)])
    for _, d in client_dirs.items():
        os.makedirs(d, exist_ok=True)
    train_setup(Path(flags.output), client_dirs, fake_pyfhel)

    if flags.fit:
        def make_client(cid):
            return train_client_factory(cid, fake_pyfhel=fake_pyfhel, client_dir=client_dirs[cid], **paths_from_flags(flags, cid))
        strategy, num_rounds = train_strategy_factory(Path("."), prefix=flags.prefix, fake_pyfhel=not flags.pyfhel)
    elif flags.predict:
        def make_client(cid):
            return test_client_factory(cid, fake_pyfhel=fake_pyfhel, client_dir=client_dirs[cid], **paths_from_flags(flags, cid))
        strategy, num_rounds = test_strategy_factory(Path("."), prefix=flags.prefix, fake_pyfhel=not flags.pyfhel)

    fl.simulation.start_simulation(
        client_fn=make_client,
        num_clients=flags.clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy)

if __name__ == "__main__":
    main()