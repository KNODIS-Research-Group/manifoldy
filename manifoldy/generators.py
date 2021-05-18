import multiprocessing
import os
import pickle
from itertools import product

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group

from manifoldy.definitions import (
    RANDOM_NOISE_STD,
    GRID,
    TARGET_DIMENSIONALITY,
    USED_CURVATURES,
    CURVATURES,
    DIFFICULTY,
    RANDOM_SEED,
)
from manifoldy.utils import get_instance_name, setup_multiprocessing


def create_curve(
    curvature_function, x0=(0, 0), v0=(1, 1), t_min=-1, t_max=1, t_step=0.01
):
    """
    Returns a function that will apply a single curvature to its arguments.
    :param curvature_function: callable. Will apply the curvature itself.
    # TODO: explanations for the rest
    :param x0:
    :param v0:
    :param t_min:
    :param t_max:
    :param t_step:
    """
    t_num = int((t_max - t_min) / t_step)
    xis = np.linspace(t_min, t_max, 10 * t_num)
    theta_fun = np.vectorize(lambda x: quad(curvature_function, 0, x)[0])
    theta_val = theta_fun(xis)
    sp_theta = CubicSpline(xis, theta_val)

    def _tang_x(xi):
        thet = sp_theta(xi)
        return v0[0] * np.cos(thet) - v0[1] * np.sin(thet)

    def _tang_y(xi):
        thet = sp_theta(xi)
        return v0[0] * np.sin(thet) + v0[1] * np.cos(thet)

    si = np.linspace(t_min, t_max, 10 * t_num)

    res_x_fun = np.vectorize(lambda x: quad(_tang_x, 0, x)[0] + x0[0])
    res_y_fun = np.vectorize(lambda x: quad(_tang_y, 0, x)[0] + x0[1])

    res_x_val = res_x_fun(si)
    res_y_val = res_y_fun(si)

    sp_res_x = CubicSpline(si, res_x_val)
    sp_res_y = CubicSpline(si, res_y_val)

    def _gamma(s):
        return np.array([sp_res_x(s), sp_res_y(s)])

    return _gamma


def get_curvature_function(curvature_types, args, n, t_min=-1, t_max=1, t_step=0.01):
    """
    Generates a curvature function, that will apply a number of curvatures to its arguments.
    :param curvature_types: list of str containing keys of CURVATURES. Individual curvature functions to be used.
    :param args: list of arguments for every curvature function.
    :param n: target dimensionality.
    #TODO: t_min, t_max, t_step
    :param t_min:
    :param t_max:
    :param t_step:
    """
    curvatures = [
        CURVATURES[curvature_types[i]](args[i]) for i in range(len(curvature_types))
    ]

    individual_curvature_functions = [
        create_curve(curvature, t_min=t_min, t_max=t_max, t_step=t_step)
        for curvature in curvatures
    ]

    def zero_pad(arg, start, end=n):
        return np.array([0] * start + list(arg) + [0] * (end - start - 2))

    def _X(arg):
        return sum(
            [
                zero_pad(individual_curvature_functions[i](arg[i]), i)
                for i in range(len(curvatures))
            ]
        )

    return _X


def create_dataset(curvature_types, args, n, std):
    """
    Generates a Phi function that applies a number of curvatures to its arguments,
    increasing its dimensionality to n.
    The result will then be rotated randomly and applied a random translation.
    :param curvature_types: a list of str with the curvatures to use. Values in this list must be found in CURVATURES.
    :param args: a list of arguments to supply for every curvature type.
    :param n: target dimensionality.
    :param std: standard deviation to apply to the random_translation.
    """
    if len(curvature_types) > n - 1:
        raise ValueError(
            "Too many curvature functions have been supplied. "
            "The resulting manifold dimensionality would exceed the target dimensionality."
        )

    curvature_function = get_curvature_function(curvature_types, args, n)
    covariance = np.eye(n) * std ** 2

    def phi(x):
        random_rotation_matrix = special_ortho_group.rvs(n, random_state=RANDOM_SEED)
        random_translation = multivariate_normal.rvs(
            mean=np.zeros(n), cov=covariance, random_state=RANDOM_SEED
        )
        return np.dot(
            random_rotation_matrix, curvature_function(x) + random_translation.T
        ).T

    return phi


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
        GRID,
    )


if __name__ == "__main__":

    setup_multiprocessing()

    with parallel_backend("loky"):
        print("Generating problem instances...")
        results = Parallel(n_jobs=-1)(
            delayed(gen_pair)(x)
            for x in product(USED_CURVATURES, USED_CURVATURES, DIFFICULTY, DIFFICULTY)
        )
        names = [x[0] for x in results]
        data = np.stack([x[1] for x in results])
        with open("results/dataset_names.pickle", "wb") as file:
            pickle.dump(names, file)
        np.save("results/dataset.npy", data)
