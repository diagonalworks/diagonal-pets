import io
import struct
import tensorflow as tf
import numpy as np
from pathlib import Path

import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitIns, Status, Code
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.client import Client

import diagonal_pets

DAYS = 64
FIT_START_DAY = 0
FIT_STOP_DAY = 56

FIT_STATE_COUNT_VISITS = 0
FIT_STATE_AGGREGATE_VISITS = 1
FIT_STATE_AGGREGATE_INFECTED_VISITS = 2
FIT_STATE_TRAIN = 3

FIT_ROUND_COUNT_VISITS = 1
FIT_ROUND_AGGREGATE_VISITS = 2
FIT_ROUND_AGGREGATE_ONE_DAY_INFECTED_VISITS = 3
FIT_ROUND_TRAIN = FIT_ROUND_AGGREGATE_ONE_DAY_INFECTED_VISITS + (FIT_STOP_DAY - FIT_START_DAY) + 1

FIT_ROUNDS = FIT_ROUND_TRAIN + 6 + 1

def fit_round_to_state(round):
    if round == 0:
        raise ValueError("Bad round")
    if round == FIT_ROUND_COUNT_VISITS:
        return FIT_STATE_COUNT_VISITS, 0
    if round == FIT_ROUND_AGGREGATE_VISITS:
        return FIT_STATE_AGGREGATE_VISITS, 0
    if round >= FIT_ROUND_AGGREGATE_ONE_DAY_INFECTED_VISITS and round < FIT_ROUND_TRAIN:
        return FIT_STATE_AGGREGATE_INFECTED_VISITS, round - FIT_ROUND_AGGREGATE_ONE_DAY_INFECTED_VISITS
    return FIT_STATE_TRAIN, round - FIT_ROUND_TRAIN + 1

class FitStrategy(Strategy):

    def __init__(self, h, ctxt, strategy, server_dir, prefix):
        self.h = h
        self.ctxt = ctxt
        diagonal_pets.init_pyfhel(self.h)
        self.strategy = strategy
        self.server_dir = server_dir
        self.prefix = prefix

    def initialize_parameters(self, client_manager):
        pass
    
    def configure_fit(self, server_round, parameters, client_manager):
        state, n = fit_round_to_state(server_round)
        print("configure_fit: round: ", server_round, " state: ", state, " n: ", n)
        if state != FIT_STATE_TRAIN:
            return [(client, FitIns(parameters, {"round": server_round})) for client in client_manager.all().values()]
        else:            
            if n == 1:
                model = diagonal_pets.make_model()
                parameters = ndarrays_to_parameters(model.get_weights())
            configuration = self.strategy.configure_fit(n, parameters, client_manager)
            for client, ins in configuration:
                ins.config["round"] = server_round
            return configuration

    def aggregate_fit(self, server_round, results, failures):
        if len(failures) > 0:
            raise ValueError("aggregate_fit: failure")
        state, n = fit_round_to_state(server_round)        
        print("aggregate_fit: round: ", server_round, " state: ", state, "n: ", n, " results:", " / ".join(["%s %s %d" % (result.status.message, result.parameters.tensor_type, sum([len(t) for t in result.parameters.tensors])) for (client, result) in results]))
        print("aggregate_fit: failures: ", failures)
        if state == FIT_STATE_COUNT_VISITS:
            return Parameters(tensors=[], tensor_type=""), {}
        elif state == FIT_STATE_AGGREGATE_VISITS:
            return self._aggregate_visits(results)
        elif state == FIT_STATE_AGGREGATE_INFECTED_VISITS:
            return self._aggregate_infected_visits(n, results)
        elif state == FIT_STATE_TRAIN:
            return self.strategy.aggregate_fit(n, results, failures)
        raise ValueError("Bad state %d" % state)

    def _aggregate_visits(self, results):
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        for client, result in results:
            aggregator.apply_fl_parameters(result.parameters, visits=True)
        parameters = Parameters(tensors=[], tensor_type="")
        aggregator.fill_fl_parameters(parameters, visits=True)
        return parameters, {}

    def _aggregate_infected_visits(self, day, results):
        parameters = Parameters(tensors=[], tensor_type="")
        if day < FIT_STOP_DAY:
            aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
            for client, result in results:
                aggregator.apply_fl_parameters(result.parameters, day=day)
            aggregator.fill_fl_parameters(parameters, day=day)
        return parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        state, n = fit_round_to_state(server_round)        
        if state == FIT_STATE_TRAIN:
            return self.strategy.configure_evaluate(n, parameters, client_manager)
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        state, n = fit_round_to_state(server_round)        
        if state == FIT_STATE_TRAIN:
            return self.strategy.aggregate_evaluate(n, results, failures)
        return None, {}

    def evaluate(self, server_round, parameters):
        if server_round > 0: # TODO: Why does this start at 0?
            state, n = fit_round_to_state(server_round)        
            if state == FIT_STATE_TRAIN:
                model = diagonal_pets.make_model()
                model.set_weights(parameters_to_ndarrays(parameters))
                model.save(self.server_dir / ("%s_model.%d.ckpt" % (self.prefix, n)))
                return self.strategy.evaluate(n, parameters)
        return None

class FitClient(Client):

    def __init__(self, id, h, ctxt, client_path, **args):
        self.id = id
        self.h = h
        self.ctxt = ctxt
        self.client_path = client_path
        self.data = diagonal_pets.make_file_data(**args)
        diagonal_pets.init_pyfhel(self.h)

    def fit(self, ins):
        state, n = fit_round_to_state(ins.config["round"])
        print("s -> %s: state: %d n: %d %s %d" % (self.id, state, n, ins.parameters.tensor_type, sum([len(t) for t in ins.parameters.tensors])))
        if state == FIT_STATE_COUNT_VISITS:
            result = self._count_visits()
        elif state == FIT_STATE_AGGREGATE_VISITS:
            result = self._aggregate_visits()
        elif state == FIT_STATE_AGGREGATE_INFECTED_VISITS:
            self._write_previous_day_aggregate(n, ins.parameters)
            result = self._aggregate_infected_visits(n)
        elif state == FIT_STATE_TRAIN:
            result = self._train(ins.parameters)
        print("%s -> s: round: %d state: %d n: %d %s %d" % (self.id, ins.config["round"], state, n, result.parameters.tensor_type, sum([len(t) for t in result.parameters.tensors])))
        return result

    def _count_visits(self):
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        infected, person_activities = diagonal_pets.read(self.data, str(self.id))
        visits, infected_visits = diagonal_pets.count_visits(infected, person_activities, FIT_START_DAY, FIT_STOP_DAY, diagonal_pets.track)            
        diagonal_pets.aggregate(visits, infected_visits, aggregator, self.h, diagonal_pets.track)
        parameters = Parameters(tensors=[], tensor_type="")
        aggregator.write(self.client_path)
        return fl.common.FitRes(Status(Code.OK, "ok"), parameters, 0, {})

    def _aggregate_visits(self):        
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        aggregator.read(self.client_path, visits=True)
        parameters = Parameters(tensors=[], tensor_type="")
        aggregator.fill_fl_parameters(parameters, visits=True)
        return fl.common.FitRes(Status(Code.OK, "ok"), parameters, 0, {})

    def _write_previous_day_aggregate(self, current_day, parameters):
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        if current_day == 0:
            aggregator.apply_fl_parameters(parameters, visits=True)
            aggregator.write(self.client_path, visits=True)
        else:
            aggregator.apply_fl_parameters(parameters, day=current_day-1)
            aggregator.write(self.client_path, day=current_day-1)

    def _aggregate_infected_visits(self, day):
        parameters = Parameters(tensors=[], tensor_type="")
        if day < FIT_STOP_DAY: # The last aggregation round only runs to save the last day's infected visits
            aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
            aggregator.read(self.client_path, day=day)
            parameters = Parameters(tensors=[], tensor_type="")
            aggregator.fill_fl_parameters(parameters, day=day)
        return fl.common.FitRes(Status(Code.OK, "ok"), parameters, 0, {})

    def _train(self, parameters):
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        aggregator.read(self.client_path)
        infected, person_activities = diagonal_pets.read(self.data, str(self.id))
        selected_people = np.add.reduce(person_activities, axis=1) > 0
        model = diagonal_pets.make_model()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.set_weights(parameters_to_ndarrays(parameters))
        events = list(diagonal_pets.sample_events(infected, selected_people, FIT_START_DAY, FIT_STOP_DAY, diagonal_pets.track))
        print("client", self.id, "events:", len(events))
        def prepared():
            examples = diagonal_pets.examples(events, person_activities, aggregator, self.h, self.ctxt, diagonal_pets.dont_track)
            return diagonal_pets.prepare_all(examples)
        input = tf.data.Dataset.from_generator(prepared, output_signature=diagonal_pets.prepared_signature)
        model.fit(x=input, epochs=1)
        return fl.common.FitRes(Status(Code.OK, "ok"), ndarrays_to_parameters(model.get_weights()), len(events), {})
