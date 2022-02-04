import os
import urllib.request
from typing import Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file


def download_from_libsvm(filename: str, dataset_path: str) -> None:
    request_url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{filename}"
    print(f"Download from {request_url}")
    urllib.request.urlretrieve(request_url, f"{dataset_path}/{filename}")


def get_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = load_svmlight_file(path)
    return data[0].toarray().astype(int), data[1]


def load_file(filename: str, n_samples: int, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(2)
    if not os.path.isfile(f"{dataset_path}/{filename}"):
        download_from_libsvm(filename, dataset_path)

    x_data, y_data = get_data(f"../dataset/{filename}")
    y_data = np.where(y_data == 1, 1, 0)

    columns = [*range(x_data.shape[1])]
    columns += [f"Not {x}" for x in columns]

    x_data = np.concatenate(
        [x_data, 1 - x_data],
        axis=1
    )


    use_idx = np.random.choice(x_data.shape[0], n_samples)
    x_data = x_data[use_idx, :]
    y_data = y_data[use_idx].astype(float)

    return x_data, y_data, columns


if __name__ == "__main__":
    download_from_libsvm("a2a", "../dataset")
