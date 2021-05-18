import multiprocessing
import os
from itertools import product

import numpy as np
from joblib import Parallel, delayed, parallel_backend

from manifoldy.utils import get_instance_name
from manifoldy.generators import gen_pair
from manifoldy.L2NormCurvature import L2_norm_sectional_curvature
from manifoldy.definitions import (
    RANDOM_NOISE_STD,
    TARGET_DIMENSIONALITY,
    USED_CURVATURES,
    DIFFICULTY,
    DIMENSIONALITY_REDUCTION_MODELS,
)
from manifoldy.test.generators_before_refactor import (
    create_dataset as create_dataset_refactor,
)
from manifoldy.test.L2NormCurvature_before_refactor import (
    L2_norm_sectional_curvature as L2_norm_sectional_curvature_refactor,
)

grid = np.mgrid[0:1:30j, 0:1:30j].reshape(2, -1).T


def gen_pair_refactor(instance):
    os.system(
        "taskset -cp 0-%d %s > /dev/null 2>&1"
        % (multiprocessing.cpu_count(), os.getpid())
    )
    name = get_instance_name(instance)
    return name, np.apply_along_axis(
        create_dataset_refactor(
            [instance[0], instance[1]],
            (instance[2], instance[3]),
            TARGET_DIMENSIONALITY,
            RANDOM_NOISE_STD,
        ),
        1,
        grid,
    )


def aux_test_gen_pair(instance):
    name, data = gen_pair(instance)
    name_r, data_r = gen_pair_refactor(instance)
    print(type(name), type(name_r))
    assert name == name_r, (name, name_r)
    print(type(data), type(data_r))
    assert np.array_equal(data, data_r), (name, name_r, data, data_r)


def test_gen_pair():
    with parallel_backend("loky"):
        print("Generating problem instances...")
        Parallel(n_jobs=-1)(
            delayed(aux_test_gen_pair)(x)
            for x in product(USED_CURVATURES, USED_CURVATURES, DIFFICULTY, DIFFICULTY)
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
            metric_estimation="interpolate_metric",
            verbose=False,
        ),
    )


def eval_pair_refactor(p, model):
    os.system(
        "taskset -cp 0-%d %s > /dev/null 2>&1"
        % (multiprocessing.cpu_count(), os.getpid())
    )
    return (
        p[0],
        type(model).__name__,
        L2_norm_sectional_curvature_refactor(
            grid,
            model.fit_transform(p[1]),
            metric_estimation="interpolate_metric",
            verbose=False,
        ),
    )


def aux_test_eval_pair(p, model):
    _, _, original = eval_pair(p, model)
    _, _, refactor = eval_pair_refactor(p, model)
    print(p[0], type(model).__name__, original, refactor, original == refactor)
    assert original == refactor


def test_eval_pair():
    with parallel_backend("loky"):
        print("Generating problem instances...")
        results = Parallel(n_jobs=-1)(
            delayed(gen_pair)(x)
            for x in product(USED_CURVATURES, USED_CURVATURES, DIFFICULTY, DIFFICULTY)
        )
        instance = dict(results)
        print("Evaluating")
        Parallel(n_jobs=-1)(
            delayed(aux_test_eval_pair)(x, y)
            for x, y in product(instance.items(), DIMENSIONALITY_REDUCTION_MODELS)
        )
