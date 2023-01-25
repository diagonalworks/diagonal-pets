import io
import struct
import tensorflow as tf
import numpy as np

import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitIns, EvaluateIns, EvaluateRes, GetParametersRes, Status, Code
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
FIT_STATE_SAVE_MODEL = 4

FIT_TRAINING_ROUNDS = 2
FIT_DAYS_PER_AGGREGATION_STEP = 4

FIT_ROUND_COUNT_VISITS = 1
FIT_ROUND_AGGREGATE_VISITS = 2
FIT_ROUND_AGGREGATE_INFECTED_VISITS = 3
FIT_ROUND_TRAIN = FIT_ROUND_AGGREGATE_INFECTED_VISITS + int((FIT_STOP_DAY - FIT_START_DAY) / FIT_DAYS_PER_AGGREGATION_STEP) + 1
FIT_ROUND_SAVE_MODEL = FIT_ROUND_TRAIN + FIT_TRAINING_ROUNDS

FIT_ROUNDS = FIT_ROUND_SAVE_MODEL
TEST_ROUNDS = 1

def fit_round_to_state(round):
    if round == 0:
        raise ValueError("Bad round")
    if round == FIT_ROUND_COUNT_VISITS:
        return FIT_STATE_COUNT_VISITS, 0
    if round == FIT_ROUND_AGGREGATE_VISITS:
        return FIT_STATE_AGGREGATE_VISITS, 0
    if round >= FIT_ROUND_AGGREGATE_INFECTED_VISITS and round < FIT_ROUND_TRAIN:
        return FIT_STATE_AGGREGATE_INFECTED_VISITS, (round - FIT_ROUND_AGGREGATE_INFECTED_VISITS) * FIT_DAYS_PER_AGGREGATION_STEP
    if round < FIT_ROUND_SAVE_MODEL:
        return FIT_STATE_TRAIN, round - FIT_ROUND_TRAIN + 1
    return FIT_STATE_SAVE_MODEL, 0

class FitStrategy(Strategy):

    def __init__(self, h, ctxt, strategy, prefix):
        self.h = h
        self.ctxt = ctxt
        diagonal_pets.init_pyfhel(self.h)
        self.strategy = strategy
        self.prefix = prefix

    def initialize_parameters(self, client_manager):
        pass

    def configure_fit(self, server_round, parameters, client_manager):
        state, n = fit_round_to_state(server_round)
        print("configure_fit: round: ", server_round, " state: ", state, " n: ", n)
        if state < FIT_STATE_TRAIN:
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
        elif state == FIT_STATE_SAVE_MODEL:
            return Parameters(tensors=[], tensor_type=""), {}
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
                aggregator.apply_fl_parameters(result.parameters, days=range(day, day + FIT_DAYS_PER_AGGREGATION_STEP))
            aggregator.fill_fl_parameters(parameters, days=range(day, day + FIT_DAYS_PER_AGGREGATION_STEP))
        return parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        state, n = fit_round_to_state(server_round)
        if server_round == FIT_ROUNDS:
            return [(client, FitIns(parameters, {"round": server_round})) for client in client_manager.all().values()]
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

class FitClient(Client):

    def __init__(self, id, h, ctxt, **args):
        print("FitClient: ", id)
        self.id = id
        self.h = h
        self.ctxt = ctxt
        self.data = diagonal_pets.make_file_data(**args)
        diagonal_pets.init_pyfhel(self.h)

    def get_parameters(self, ins):
        parameters = Parameters(tensors=[], tensor_type="")
        return GetParametersRes(Status(Code.OK, "ok"), parameters)

    def evaluate(self, ins):
        return EvaluateRes(Status(Code.OK, "ok"), 0.0, 0, {})

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
        elif state == FIT_STATE_SAVE_MODEL:
            result = self._save_model(ins.parameters)
        print("%s -> s: round: %d state: %d n: %d %s %d" % (self.id, ins.config["round"], state, n, result.parameters.tensor_type, sum([len(t) for t in result.parameters.tensors])))
        return result

    def _count_visits(self):
        infected, person_activities = diagonal_pets.read(self.data, str(self.id))
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        visits, infected_visits = diagonal_pets.count_visits(infected, person_activities, FIT_START_DAY, FIT_STOP_DAY, diagonal_pets.track)
        diagonal_pets.aggregate_and_write(visits, infected_visits, aggregator, self.data.aggregator_directory(), diagonal_pets.track)
        parameters = Parameters(tensors=[], tensor_type="")
        return fl.common.FitRes(Status(Code.OK, "ok"), parameters, 0, {})

    def _aggregate_visits(self):
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        aggregator.read(self.data.aggregator_directory(), visits=True)
        parameters = Parameters(tensors=[], tensor_type="")
        aggregator.fill_fl_parameters(parameters, visits=True)
        return fl.common.FitRes(Status(Code.OK, "ok"), parameters, 0, {})

    def _write_previous_day_aggregate(self, current_day, parameters):
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        if current_day == 0:
            aggregator.apply_fl_parameters(parameters, visits=True)
            aggregator.write(self.data.aggregator_directory(), visits=True)
        else:
            aggregator.apply_fl_parameters(parameters, days=range(current_day - FIT_DAYS_PER_AGGREGATION_STEP, current_day))
            aggregator.write(self.data.aggregator_directory(), days=range(current_day - FIT_DAYS_PER_AGGREGATION_STEP, current_day))

    def _aggregate_infected_visits(self, day):
        parameters = Parameters(tensors=[], tensor_type="")
        if day < FIT_STOP_DAY: # The last aggregation round only runs to save the last day's infected visits
            aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
            aggregator.read(self.data.aggregator_directory(), days=range(day, day+FIT_DAYS_PER_AGGREGATION_STEP))
            parameters = Parameters(tensors=[], tensor_type="")
            aggregator.fill_fl_parameters(parameters, days=range(day, day+FIT_DAYS_PER_AGGREGATION_STEP))
        return fl.common.FitRes(Status(Code.OK, "ok"), parameters, 0, {})

    def _train(self, parameters):
        aggregator = diagonal_pets.make_aggregator(self.h, self.ctxt)
        aggregator.read(self.data.aggregator_directory(), visits=True)
        infected, person_activities = diagonal_pets.read(self.data, str(self.id))
        model = diagonal_pets.make_model()
        model.set_weights(parameters_to_ndarrays(parameters))
        n = diagonal_pets.fit(infected, person_activities, model, aggregator, self.data, diagonal_pets.track, epochs=1)
        return fl.common.FitRes(Status(Code.OK, "ok"), ndarrays_to_parameters(model.get_weights()), n, {})

    def _save_model(self, parameters):
        model = diagonal_pets.make_model()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.set_weights(parameters_to_ndarrays(parameters))
        model.save(self.data.model_filename())
        return fl.common.FitRes(Status(Code.OK, "ok"), parameters, 1, {})

class TestClient(Client):

    def __init__(self, id, h, **args):
        self.id = id
        self.h = h
        self.data = diagonal_pets.make_file_data(**args)
        diagonal_pets.init_pyfhel(self.h)

    def get_parameters(self, ins):
        parameters = Parameters(tensors=[], tensor_type="")
        return GetParametersRes(Status(Code.OK, "ok"), parameters)

    def evaluate(self, ins):
        diagonal_pets.predict_parallel(self.h, self.data)
        return EvaluateRes(Status(Code.OK, "ok"), 0.0, 0, {})

class TestStrategy(Strategy):

    def initialize_parameters(self, client_manager):
        pass

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results, failures):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return [(client, EvaluateIns(parameters, {})) for client in client_manager.all().values()]

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None