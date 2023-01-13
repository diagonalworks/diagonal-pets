#!/usr/bin/env python3

import sys
sys.path.append("/code_execution/src/")

import argparse
import flwr as fl
import os

from pathlib import Path

import diagonal_pets

def train_strategy_factory(server_dir, fake_pyfhel=False, prefix="dw"):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    strategy = fl.server.strategy.FedAvg()
    return diagonal_pets.FitStrategy(h, ctxt, strategy, server_dir, prefix), diagonal_pets.FIT_ROUNDS

def train_client_factory(cid, fake_pyfhel=False, **args):
    h, ctxt = diagonal_pets.make_pyfhel(fake_pyfhel)
    return diagonal_pets.FitClient(cid, h, ctxt, **args)

def paths_from_flags(flags, cid):
    directory = Path(flags.input)
    return {
        "person_data_path": directory / ("%s_person.%s.csv.gz" % (flags.prefix, cid)),
        "activity_location_data_path":  directory / ("%s_activity_locations.%s.csv.gz" % (flags.prefix, cid)),
        "activity_location_assignment_data_path": directory / ("%s_activity_location_assignment.%s.csv.gz" % (flags.prefix, cid)),
        "disease_outcome_data_path": directory / ("%s_disease_outcome_training.%s.csv.gz" % (flags.prefix, cid)),
        "model_dir": Path(flags.output),
        "preds_format_path": Path(flags.output) / ("%s_disease_outcome_target.%s.csv.gz" % (flags.prefix, cid)),
        "preds_dest_path": Path(flags.output) / ("%s_predictions.%s.csv" % (flags.prefix, cid)),
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
    flags = parser.parse_args()


    client_paths = [Path(flags.output) / "client-tmp" / str(i+1) for i in range(0, flags.clients)]
    for p in client_paths:
        os.makedirs(p, exist_ok=True)
    strategy, num_rounds = train_strategy_factory(Path("."), prefix=flags.prefix, fake_pyfhel=not flags.pyfhel)
    def make_client(cid):
        return train_client_factory(cid, fake_pyfhel=not flags.pyfhel, client_path=client_paths[int(cid)-1], **paths_from_flags(flags, cid))
    fl.simulation.start_simulation(
        client_fn=make_client,
        num_clients=flags.clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy)

if __name__ == "__main__":
    main()