from itertools import combinations
from typing import List, Tuple, Iterator

import numpy as np
from joblib import Parallel, delayed
from scipy import special
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
from loguru import logger


class FeatureSelection:

    def __init__(
            self,
            train_dataset: np.ndarray, train_target: np.ndarray, num_select_features: int,
            num_select_combs: int, num_multiply: int
    ) -> None:
        self.train_dataset: np.ndarray = train_dataset
        self.train_target: np.ndarray = train_target

        self.num_rows: int = self.train_dataset.shape[0]
        self.num_features: int = self.train_dataset.shape[1]

        self.num_select_features: int = num_select_features
        self.num_select_combs: int = num_select_combs

        self.num_multiply: int = num_multiply

    def process(self, combs) -> List:
        mut_x: np.ndarray = np.prod(self.train_dataset[:, list(combs)], axis=1).reshape(-1, 1).astype(bool)

        mut_info_value: float = mutual_info_classif(
            mut_x, self.train_target, n_neighbors=2, discrete_features=True
        )[0]  # 特徴数は1しかないので
        # print(comb, mut_info_value)

        # if mut_info_value > 0:
        return [mut_info_value, combs]

    def selection(self, combs_iter: Iterator, iter_size: int, num_selection: int, n_jobs: int = -1) -> List[Tuple]:
        results = Parallel(n_jobs=n_jobs)(delayed(self.process)(i) for i in tqdm(combs_iter, total=iter_size))
        results = sorted(results, reverse=True, key=lambda x: x[0])

        use_combs_fea = []
        last_score = 0
        for idx, (score, combs) in enumerate(results):
            if last_score != score:
                use_combs_fea.append(combs)
                last_score = score
                if len(use_combs_fea) == num_selection:
                    break
        return use_combs_fea
        # return list(map(lambda x: x[1], results[:num_selection]))

    def run_selection(self) -> List[Tuple]:
        logger.info("Feature selection")
        base_features_iter = out_all_comb(self.num_features, 1)
        base_features: List[Tuple] = self.selection(
            combs_iter=base_features_iter,
            iter_size=self.num_features,
            num_selection=self.num_select_features,
            n_jobs=-1
        )
        print(base_features)

        logger.info("Combination feature selection")
        n_columns = len(base_features)

        base_fea = [x[0] for x in base_features]
        # combs_iter = out_all_comb(len(base_fea), self.num_multiply)
        combs_iter = second_comb(base_fea, self.num_multiply)
        iter_size = sum(special.comb(n_columns, i + 1, exact=True) for i in range(self.num_multiply))
        combs_features: List[Tuple] = self.selection(
            combs_iter=combs_iter,
            iter_size=iter_size,
            num_selection=self.num_select_combs,
            n_jobs=-1
        )
        return combs_features


def out_all_comb(num, max_num):
    for i in range(1, max_num + 1):
        for cb in combinations(range(num), i):
            yield cb


def second_comb(base_fea, max_num):
    for i in range(1, max_num + 1):
        for cb in combinations(base_fea, i):
            yield cb