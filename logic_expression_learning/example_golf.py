from feature_selection import FeatureSelection
from load_dataset import get_data
from runner import SingleRunner
from utils import *


def run(cfg):
    print(cfg)
    x_data, y_data, columns = get_data(
        dataset_name=cfg.dataset_name,
        n_samples=cfg.num_rows,
        cfg=cfg
    )

    y_data = y_data * 2 - np.ones_like(y_data)

    feaselect = FeatureSelection(
        train_dataset=x_data,
        train_target=y_data,
        num_select_features=cfg.num_select_features,
        num_select_combs=cfg.num_select_combinations,
        num_multiply=cfg.num_multiply
    )
    all_comb = feaselect.run_selection()

    runner = SingleRunner(
        train_dataset=x_data,
        train_target=y_data,
        combs_fea=all_comb,
        poly_weight=len(all_comb),
        norm_weight=cfg.norm_weight,
        max_combs=cfg.max_combs,
        timeout=cfg.timeout
    )

    runner.solve()


initialize_cfg = {
    "dataset_path": "../dataset/",
    "dataset_name": "golf",
    "timeout": 500,
    "seed": 42,
    "num_rows": 1000,
    "num_select_features": 30,
    "num_select_combinations": 4000,
    "norm_weight": 3,
    "num_multiply": 3,
    "max_combs": 20
}

config = OmegaConf.create(initialize_cfg)
run(config)
