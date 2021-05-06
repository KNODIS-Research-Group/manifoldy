#!/usr/bin/env python
# coding: utf-8

import multiprocessing
import os
import pickle
from itertools import product

import numpy as np
from joblib import Parallel, delayed, parallel_backend

from manifoldy.definitions import DIMENSIONALITY_REDUCTION_MODELS
from manifoldy.utils import setup_multiprocessing


def apply_reduction(instance, model):
    name, instance_data = instance
    os.system(
        "taskset -cp 0-%d %s > /dev/null 2>&1"
        % (multiprocessing.cpu_count(), os.getpid())
    )
    return name, type(model).__name__, model.fit_transform(instance_data)


if __name__ == "__main__":
    setup_multiprocessing()

    with open("results/dataset_names.pickle", "rb") as file:
        names = pickle.load(file)

    data = np.load("results/dataset.npy")

    instances = zip(names, data)

    with parallel_backend("loky"):
        print("Applying dimensionality reduction algorithms...")
        results = Parallel(n_jobs=-1)(
            delayed(apply_reduction)(instance, model)
            for instance, model in product(instances, DIMENSIONALITY_REDUCTION_MODELS)
        )

        projection_names = [x[0] + " " + x[1] for x in results]
        projection_data = [x[2] for x in results]

        with open("results/projection_dataset_names.pickle", "wb") as file:
            pickle.dump(projection_names, file)
        np.save("results/projection_results.npy", projection_data)
