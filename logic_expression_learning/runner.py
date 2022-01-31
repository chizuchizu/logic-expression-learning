import json
import os
from typing import List, Tuple

import amplify
import numpy as np
from amplify import Solver, BinarySymbolGenerator, BinaryQuadraticModel
from amplify.client import FixstarsClient
from amplify.constraint import equal_to, greater_equal, less_equal, penalty
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import roc_auc_score

from utils import get_poly_from_bits, predict, normalize


class Runner:

    def __init__(self, train_dataset: np.ndarray, train_target: np.ndarray, combs_fea: List[Tuple],
                 D_weight: np.ndarray,
                 poly_weight: float = 1,
                 norm_weight: float = 0,
                 max_combs: int = 10,
                 timeout: int = 3000,
                 ) -> None:
        self.solver: amplify.Solver = define_solver(timeout)
        self.b_symbol_gen: amplify.BinarySymbolGenerator = BinarySymbolGenerator()

        self.train_dataset: np.array = train_dataset
        self.train_target: np.array = train_target

        self.D = D_weight

        self.num_rows: int = self.train_dataset.shape[0]
        self.num_features: int = self.train_dataset.shape[1]
        self.combs_fea: List[Tuple] = combs_fea
        self.length_combs: int = len(self.combs_fea)

        self.poly_weight: float = poly_weight
        # self.norm_weight: float = norm_weight

        self.max_combs: int = max_combs
        self.norm_weight: float = norm_weight

        self.class_weight_1 = self.num_rows / (2 * self.train_target.sum())
        self.class_weight_0 = self.num_rows / (2 * (self.num_rows - self.train_target.sum()))

        self.bweight_is_use_members = self.b_symbol_gen.array(self.length_combs)
        self.num_all_bit = self.length_combs

    def build_logical_model(self) -> amplify.BinaryQuadraticModel:
        model_constraints: amplify.BinaryConstraint = 0

        logger.info("Build logical model")

        for idx in tqdm(range(self.num_rows)):
            pred = 0
            size_encoding_bit: int = get_poly_from_bits(
                data=self.train_dataset[idx, :],
                weight=[1 for _ in range(self.length_combs)],
                combs=self.combs_fea
            )
            logic_polynomial: amplify.BinaryPoly = self.build_logic_expression(idx)

            """ Order Encoding """
            encoding_bit: List[amplify.BinaryPoly] = self.b_symbol_gen.array(size_encoding_bit)
            model_constraints += apply_order_encoding(encoding_bit, logic_polynomial)
            """"""
            if size_encoding_bit >= 1:
                encoding_bit_new = (encoding_bit[0] * 2) - 1
                pred += encoding_bit_new

            """評価関数"""
            if size_encoding_bit >= 1:
                if self.train_target[idx] == 0:
                    model_constraints += equal_to(pred - self.train_target[idx], 0) * self.poly_weight * (
                        self.class_weight_0
                    ) * self.D[idx]
                else:
                    model_constraints += equal_to(pred - self.train_target[idx], 0) * self.poly_weight * (
                        self.class_weight_1
                    ) * self.D[idx]
            """"""

        # model_quadratic: amplify.BinaryQuadraticModel = model_constraints + sum(
        #    self.bweight_is_use_members) * self.norm_weight * self.num_rows
        norm_constraint_1 = less_equal(sum(self.bweight_is_use_members),
                                       self.max_combs) * self.norm_weight  # / self.num_rows * 1600
        norm_constraint = penalty(sum(self.bweight_is_use_members), le=self.max_combs) * self.norm_weight
        # model_quadratic = model_constraints + norm_constraint + self.b_symbol_gen.array(1)[
        #     0]  # +sum(self.bweight_is_use_members)
        # model_quadratic = model_constraints + sum(self.bweight_is_use_members) * self.norm_weight * self.num_rows
        #        norm_constraint = penalty(sum(self.bweight_is_use_members), le=self.max_combs) * self.norm_weight
        model_quadratic = model_constraints +  norm_constraint
        model_quadratic = BinaryQuadraticModel(model_quadratic)
        # model_quadratic = model_constraints + self.b_symbol_gen.array(1)[0]* 0.001
        return model_quadratic

    def build_logic_expression(self, idx) -> amplify.BinaryPoly:
        logic_expression: amplify.BinaryPoly = 0

        for i, member_elements in enumerate(self.combs_fea):
            comb_idx_weight: amplify.BinaryPoly = self.bweight_is_use_members[i]
            for element_idx in list(member_elements):
                # AND演算
                comb_idx_weight *= self.train_dataset[idx, element_idx]
            # OR演算（とりあえず全部足す）
            logic_expression += comb_idx_weight

        return logic_expression

    def solve(self) -> Tuple:
        penalties = self.build_logical_model()
        print(penalties.num_logical_vars)
        result: amplify.SolverResult = self.solver.solve(penalties)[0]
        client_result: amplify.client.FixstarsClientResult = self.solver.client_result

        bweight_solution: np.ndarray = parse_weight_from_result(self.num_all_bit, result)

        logical_eqs = bweight_solution[:self.length_combs]

        logger.info(f"Annealing time: {client_result.annealing_time_ms}")
        logger.info(f"time stamp: {client_result.timing.time_stamps}")
        logger.info(f"Is feasible: {result.is_feasible}")

        return logical_eqs, client_result, result


def parse_weight_from_result(max_length: int, calc_result) -> np.ndarray:
    weight = np.zeros(max_length, dtype=np.int_)
    item = calc_result.values
    for i in range(max_length):
        if item.get(i) is not None:
            weight[i] = item[i]
        else:
            weight[i] = 0

    return weight


def define_solver(timeout: int = 3000) -> amplify.Solver:
    client: amplify.client.FixstarsClient = FixstarsClient()
    token_path = os.path.expanduser("~/.amplify/token.json")

    with open(token_path) as f:
        client.token = json.load(f)["AMPLIFY_TOKEN"]

    client.parameters.timeout = timeout
    solver: amplify.Solver = Solver(client)
    solver.filter_solution = False
    client.parameters.outputs.duplicate = True
    return solver


def apply_unary_encoding(bits, poly_equal_to):
    const = equal_to(
        sum(bits) - poly_equal_to,
        0
    )

    return const


def apply_order_encoding(encoding_bits, poly_equal_to):
    const = apply_unary_encoding(encoding_bits, poly_equal_to)

    for i in range(len(encoding_bits) - 1):
        const += greater_equal(
            encoding_bits[0] - encoding_bits[i + 1],
            0
        )

    return const


class AdaBoostRunner:

    def __init__(self, train_dataset: np.ndarray, train_target: np.ndarray, combs_fea: List[Tuple],
                 poly_weight: float = 1,
                 norm_weight: float = 0,
                 max_combs: int = 10,
                 timeout: int = 3000,
                 ):
        self.timeout = timeout

        self.train_dataset: np.array = train_dataset
        self.train_target: np.array = train_target

        self.num_rows, self.num_features = self.train_dataset.shape
        self.combs_fea: List[Tuple] = combs_fea
        self.length_combs: int = len(self.combs_fea)

        # set initial value
        self.D = np.ones(self.num_rows) / self.num_rows

        # self.poly_weight: float = poly_weight
        self.norm_weight: float = norm_weight
        self.max_combs: int = max_combs

        self.expression_weights = []
        self.expressions = []

        self.epsilon_list = []

    def solve(self, is_logging_result=True):
        runner = Runner(
            train_dataset=self.train_dataset,
            train_target=self.train_target,
            combs_fea=self.combs_fea,
            D_weight=self.D,
            poly_weight=self.length_combs,
            norm_weight=self.norm_weight,
            max_combs=self.max_combs,
            timeout=self.timeout
        )
        logical_eqs, client_result, result = runner.solve()

        for i, comb in enumerate(logical_eqs):
            if comb:
                print(self.combs_fea[i])

        # update D, wj, logical_expression
        is_skip_loop = self.update_parameters(logical_eqs)
        if is_skip_loop:
            return

        if is_logging_result:
            self.write_metric(client_result, result)

        return logical_eqs

    def predict_local(self, dataset, logical_expression):
        prediction = np.array(predict(dataset, logical_expression, self.combs_fea))
        return prediction

    def predict_global(self, dataset):
        prediction = np.zeros(dataset.shape[0])

        for i_exp, i_exp_weight in zip(self.expressions, self.expression_weights):
            pred_i = self.predict_local(dataset, i_exp) * i_exp_weight
            prediction += pred_i

        return prediction

    def update_parameters(self, logical_expression):
        train_prediction = self.predict_local(self.train_dataset, logical_expression)

        epsilon = (self.D * (train_prediction != self.train_target)).sum()  # / self.D.sum()
        wj = np.log((1 - epsilon) / epsilon) / 2
        print((train_prediction == self.train_target).mean())

        tp = normalize(train_prediction.astype(float))
        tt = normalize(self.train_target.astype(float))
        try:
            print("AUC", roc_auc_score(tt, tp))
        except:
            print("error"
                  "")

        if wj < 0:
            return True
        # self.D = self.D * np.exp(wj * (train_prediction != self.train_target))
        self.D = self.D * np.exp(-wj * train_prediction * self.train_target)
        self.D /= self.D.sum()

        self.expressions.append(logical_expression)
        self.expression_weights.append(wj)
        self.epsilon_list.append(epsilon)
        return False

    def get_auc_score(self, dataset, target, is_target_normalize=True):
        prediction = self.predict_global(dataset)
        target = normalize(target)
        if is_target_normalize:
            prediction = normalize(prediction)

        auc_score = roc_auc_score(target, prediction)
        return auc_score

    def write_metric(
            self,
            client_result,
            result
    ):
        metrics = {
            "AnnealingTime": client_result.annealing_time_ms,
            "Energy": result.energy,
            "NumIterations": client_result.execution_parameters.num_iterations,
            "global_TrainAUC": self.get_auc_score(self.train_dataset, self.train_target),
            "Epsilon": self.epsilon_list[-1], "wj": self.expression_weights[-1]
        }

        logger.info(f"{metrics}")


class SingleRunner:

    def __init__(self, train_dataset: np.ndarray, train_target: np.ndarray, combs_fea: List[Tuple],
                 poly_weight: float = 1,
                 norm_weight: float = 0,
                 max_combs: int = 10,
                 timeout: int = 3000,
                 ):
        self.timeout = timeout

        self.train_dataset: np.array = train_dataset
        self.train_target: np.array = train_target

        self.num_rows, self.num_features = self.train_dataset.shape
        self.combs_fea: List[Tuple] = combs_fea
        self.length_combs: int = len(self.combs_fea)

        # self.poly_weight: float = poly_weight
        self.norm_weight: float = norm_weight
        self.max_combs: int = max_combs

        self.expression = None

    def solve(self, is_logging_result=True):
        runner = Runner(
            train_dataset=self.train_dataset,
            train_target=self.train_target,
            combs_fea=self.combs_fea,
            D_weight=np.ones(self.num_rows) / self.num_rows,
            poly_weight=self.length_combs,
            norm_weight=self.norm_weight,
            max_combs=self.max_combs,
            timeout=self.timeout
        )
        logical_eqs, client_result, result = runner.solve()

        for i, comb in enumerate(logical_eqs):
            if comb:
                print(self.combs_fea[i])
        self.expression = logical_eqs
        self.write_metric(client_result, result)
        return logical_eqs

    def predict(self, dataset):
        prediction = np.array(predict(dataset, self.expression, self.combs_fea))
        return prediction

    def get_auc_score(self, dataset, target, is_target_normalize=True):
        prediction = self.predict(dataset)
        target = normalize(target)
        if is_target_normalize:
            prediction = normalize(prediction)

        auc_score = roc_auc_score(target, prediction)
        return auc_score

    def write_metric(
            self,
            client_result,
            result
    ):
        metrics = {
            "AnnealingTime": client_result.annealing_time_ms,
            "Energy": result.energy,
            "NumIterations": client_result.execution_parameters.num_iterations,
            "TrainAUC": self.get_auc_score(self.train_dataset, self.train_target),
        }

        logger.info(f"{metrics}")
