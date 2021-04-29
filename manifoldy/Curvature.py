#!/usr/bin/env python
# coding: utf-8

import multiprocessing
import os
from itertools import product

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed, parallel_backend

from manifoldy.L2NormCurvature import L2_norm_sectional_curvature
from manifoldy.definitions import (
    RANDOM_NOISE_STD,
    TARGET_DIMENSIONALITY,
    USED_CURVATURES,
    DIFF,
    DIMENSIONALITY_REDUCTION_MODELS,
)
from manifoldy.generators import create_dataset

grid = np.mgrid[0:1:30j, 0:1:30j].reshape(2, -1).T
process = psutil.Process(os.getpid())
process.cpu_affinity(range(multiprocessing.cpu_count()))


def get_instance_name(instance):
    name = instance[0][0].upper() + instance[1][0].upper()
    name += (
        ("E" if instance[2] == 0.8 else "H")
        + ("E" if instance[3] == 0.8 else "H")
        + str(TARGET_DIMENSIONALITY)
    )
    return name


def gen_pair(instance):
    os.system(
        "taskset -cp 0-%d %s > /dev/null 2>&1"
        % (multiprocessing.cpu_count(), os.getpid())
    )
    name = get_instance_name(instance)
    return name, np.apply_along_axis(
        create_dataset(
            [instance[0], instance[1]],
            (instance[2], instance[3]),
            TARGET_DIMENSIONALITY,
            RANDOM_NOISE_STD,
        ),
        1,
        grid,
    )


def eval_pair(p, model):
    os.system(
        "taskset -cp 0-%d %s > /dev/null 2>&1"
        % (multiprocessing.cpu_count(), os.getpid())
    )
    return (
        p[0],
        type(model).__name__,
        L2_norm_sectional_curvature(
            grid,
            model.fit_transform(p[1]),
            metric_estimation="interpolate",
            verbose=False,
        ),
    )


def main():
    with parallel_backend("loky"):
        print("Generating problem instances...")
        results = Parallel(n_jobs=-1)(
            delayed(gen_pair)(x)
            for x in product(USED_CURVATURES, USED_CURVATURES, DIFF, DIFF)
        )
        instance = dict(results)
        print("Evaluating")
        res = Parallel(n_jobs=-1)(
            delayed(eval_pair)(x, y)
            for x, y in product(instance.items(), DIMENSIONALITY_REDUCTION_MODELS)
        )

        # create DataFrame using data
        df = pd.DataFrame(res, columns=["Instance", "Model", "Score"])

        try:
            previous_df = pd.read_csv("results.csv", index_col=0)
            if not previous_df.equals(df):
                print("Different files")
                df.to_csv("results2.csv")
            else:
                print("Files are identical")
        except FileNotFoundError:
            df.to_csv("results.csv")


if __name__ == "__main__":
    main()
