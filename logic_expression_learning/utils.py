from typing import List, Tuple

import os
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def get_poly_from_bits(data, weight, combs) -> int:
    logic_expression = 0
    for idx, comb in enumerate(combs):
        product = weight[idx]
        for x_ in comb:
            product *= data[x_]
        logic_expression += product
    return logic_expression


def check_pred_is_correct(data, weight, combs) -> int:
    logic_expression = get_poly_from_bits(data, weight, combs)
    if logic_expression == 0:
        return -1
        # return 0
    else:
        return 1


def generate_train_df(df_nrows, n_fea):
    df = pd.DataFrame(np.random.randint(2, size=(df_nrows, n_fea)).astype(int))

    target_list = []
    # print(df)
    for i in range(df_nrows):
        target = int(generate_target_based_on_formula(df.iloc[i, :]))
        target_list.append(target)

    df = pd.concat([df, 1 - df], axis=1)
    df["target"] = target_list

    return df


def generate_target_based_on_formula(x):
    # return ((not x[0]) & x[1] & (not x[2]) & (not x[3])) | (x[1] & (not x[2]) & x[3]) | (
    #        x[0] & (not x[1]) & x[2] & x[3]) | (x[0] & x[2] & x[3]) | (x[1] & x[2] & x[3])
    # return (x[0] & x[1] & x[2])#  | (x[1] & x[2] & x[3])
    # return (x[0] & x[1]) | (x[1] & x[2])
    # return x[0] | x[1] | x[2]
    return x[0] ^ x[1] ^ x[2] ^ x[3]  # ^ x[4] # ^ x[5]


def add_noise(target, add_noise_rate):
    noise_num = int(len(target) * add_noise_rate)
    target.iloc[:noise_num] = 1 - target.iloc[:noise_num]
    return target


def predict(data: np.ndarray, weight: np.ndarray, combs: List[Tuple]) -> np.ndarray:
    pred_list: List[int] = []
    for idx in range(len(data)):
        pred = check_pred_is_correct(data[idx, :], weight, combs)
        # print(f"{pred} | {target[idx]}")

        pred_list.append(pred)
    return np.array(pred_list).astype(float)


def save_log_cfg(cfg):
    save_dir = f"exp/{cfg.exp_name}"
    if os.path.isdir(save_dir):
        logger.error("The experiment is exists")
        exit()

    logger.add(f"{save_dir}/out.log")
    OmegaConf.save(cfg, f"{save_dir}/config.yaml")


def normalize(array):
    array -= array.min()
    array /= array.max()
    return array
