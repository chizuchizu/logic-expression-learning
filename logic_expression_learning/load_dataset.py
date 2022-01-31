import pandas as pd
import numpy as np

import load_LIBSVM


def get_data(
        dataset_name: str,
        n_samples: int,
        cfg=None
):
    if dataset_name == "mushroom":
        return get_mushroom(n_samples, cfg)
    elif dataset_name == "car_evaluation":
        return get_car_evaluation(n_samples, cfg)
    elif dataset_name == "golf":
        return get_golf()
    else:
        return load_LIBSVM.load_file(dataset_name, n_samples)


def get_mushroom(n_samples, cfg):
    dataset = pd.read_csv("../../LogicTreeStreamlit/src/dataset/mushrooms.csv")
    dataset = pd.get_dummies(dataset)
    dataset = dataset.sample(n=n_samples, random_state=cfg.seed)

    not_ds = 1 - dataset.iloc[:, 2:]
    not_ds.columns = map(
        lambda x: "not_" + x,
        not_ds.columns
    )

    dataset = pd.concat(
        [
            dataset,
            not_ds
        ],
        axis=1
    )

    x_data = dataset.iloc[:, 2:].values
    # if cfg.use_mushroom_columns:
    y_data = dataset.iloc[:, 0].values.astype("float64")

    return x_data, y_data, dataset.columns

def get_car_evaluation(n_samples, cfg):
    dataset = pd.read_csv("../dataset/car_evaluation.csv", header=None)
    dataset.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

    dataset = dataset.loc[:, dataset.columns[::-1]]

    dataset["class"] = np.where(dataset["class"] == "unacc", "unacc", "acc")

    dataset = pd.get_dummies(dataset)
    dataset = dataset.sample(n=n_samples, random_state=cfg.seed)

    not_ds = 1 - dataset.iloc[:, 2:]
    not_ds.columns = map(
        lambda x: "not_" + x,
        not_ds.columns
    )

    dataset = pd.concat(
        [
            dataset,
            not_ds
        ],
        axis=1
    )

    x_data = dataset.iloc[:, 2:].values
    # if cfg.use_mushroom_columns:
    y_data = dataset.iloc[:, 0].values.astype("float64")

    return x_data, y_data, dataset.columns

def get_golf():
    dataset = pd.read_csv("../dataset/golf_df.csv")

    dataset = pd.get_dummies(dataset)

    dataset = dataset.loc[:, dataset.columns[::-1]]
    not_ds = 1 - dataset.iloc[:, 2:]
    not_ds.columns = map(
        lambda x: "not_" + x,
        not_ds.columns
    )

    dataset = pd.concat(
        [
            dataset,
            not_ds
        ],
        axis=1
    )

    x_data = dataset.iloc[:, 2:].values
    # if cfg.use_mushroom_columns:
    y_data = dataset.iloc[:, 0].values.astype("float64")

    return x_data, y_data, dataset.columns

