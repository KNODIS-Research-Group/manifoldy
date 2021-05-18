#!/usr/bin/env python
# coding: utf-8

import copy
import datetime
import multiprocessing
import os
import pickle
from functools import reduce

import ndsplines
import numpy as np
import pandas as pd
from joblib import parallel_backend, Parallel, delayed
from scipy.integrate import nquad
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy.special import comb

from manifoldy.definitions import GRID


def compute_first_christoffel_symbols(dg, n):
    CS_first = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                CS_first[k, i, j] = 1 / 2 * (dg[j][k, i] + dg[i][k, j] - dg[k][i, j])

    return CS_first


def compute_partial_derivative_first_christoffel_symbols(ddg, l, n):
    CS_first_diff = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                CS_first_diff[k, i, j] = (
                    1 / 2 * (ddg[l][j][k, i] + ddg[l][i][k, j] - ddg[l][k][i, j])
                )

    return CS_first_diff


def compute_derivative_first_christoffel_symbols(ddg, n):
    derivatives = []
    for l in range(n):
        CS_first_diff = compute_partial_derivative_first_christoffel_symbols(ddg, l, n)
        derivatives.append(CS_first_diff)
    return np.stack(derivatives)


def compute_christoffel_symbols(g, dg):
    # Index 0: Upper index
    # Index 1, 2: Lower indices
    # Gamma_{1,2}^0
    n = g.shape[0]
    CS_first = compute_first_christoffel_symbols(dg, n)

    g_inv = np.linalg.inv(g)
    return np.tensordot(g_inv, CS_first, axes=([1], [0]))


def compute_derivative_christoffel_symbols(g, dg, ddg):
    # Index 0: Derivative of the symbol
    # Index 1: Upper index
    # Index 2, 3: Lower indices
    # Gamma_{0;2,3}^1
    n = g.shape[0]
    CS_first = compute_first_christoffel_symbols(dg, n)
    g_inv = np.linalg.inv(g)

    derivatives = []
    for l in range(n):
        CS_first_diff_l = compute_partial_derivative_first_christoffel_symbols(
            ddg, l, n
        )
        g_inv_diff_l = -np.matmul(np.matmul(g_inv, dg[l]), g_inv)
        CS_l = np.tensordot(g_inv_diff_l, CS_first, axes=([1], [0])) + np.tensordot(
            g_inv, CS_first_diff_l, axes=([1], [0])
        )
        derivatives.append(CS_l)

    return np.stack(derivatives)


def compute_curvature_tensor(g, dg, ddg):
    # Index 0: Upper index
    # Index 1, 2, 3: Lower indices
    # R[l,i,j,k] = R^l_{ijk} = dx^l(R(ei,ej)ek)
    n = g.shape[0]

    CS = compute_christoffel_symbols(g, dg)
    CS_diff = compute_derivative_christoffel_symbols(g, dg, ddg)
    R = np.zeros((n, n, n, n))
    CS_contract = np.tensordot(CS, CS, axes=([0], [2]))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    R[l, i, j, k] = (
                        CS_diff[j, l, i, k]
                        - CS_diff[i, l, j, k]
                        + CS_contract[i, k, l, j]
                        - CS_contract[j, k, l, i]
                    )

    return R


def compute_first_curvature_tensor(g, dg, ddg):
    # Index 0: Upper index
    # Index 1, 2, 3: Lower indices
    # R[l,i,j,k] = R^l_{ijk} = dx^l(R(ei,ej)ek)
    # n = g.shape[0]

    R = compute_curvature_tensor(g, dg, ddg)
    R_first = np.tensordot(R, g, axes=([0], [0]))

    return R_first


def compute_sectional_curvatures(g, dg, ddg):
    # Returns a vector whose entries are
    # sectional curvatures at (e1,e2), (e1,e3),...,(e1,en)
    # (e2,e3), (e2,e4), ..., (e2, en),...
    n = g.shape[0]
    R = compute_first_curvature_tensor(g, dg, ddg)
    sect_curvs = np.zeros(int(comb(n, 2)))
    counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            area = g[i, i] * g[j, j] - g[i, j] ** 2
            sect_curvs[counter] = R[i, j, i, j] / area
            counter += 1

    return np.stack(sect_curvs)


# UTILS


def dataset_to_grid(X, n_samples, dim):
    shap = n_samples + [dim]
    # shap = [dim] + n_samples
    return X.reshape(shap)


def grid_data(X, Y, n_samples=None):
    dim = X.shape[1]
    if n_samples is None:
        n_samples = [41] * dim

    offset = 0.05
    xs = np.array(
        [np.linspace(0 + offset, 1 - offset, n_samples[i]) for i in range(dim)]
    )
    X_new = np.array(np.meshgrid(*xs))

    Y_new = []
    for i in range(dim):
        grid_y = griddata(X, Y[:, i], tuple(X_new), method="linear")
        Y_new.append(grid_y)
    Y_new = np.stack(Y_new)

    n_total_samples = 1
    for n in n_samples:
        n_total_samples *= n

    return (
        X_new.reshape(dim, n_total_samples).T,
        Y_new.reshape(dim, n_total_samples).T,
        n_samples,
    )


def estimate_inner_product(X, b):
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    S = np.zeros((u.shape[0], vh.shape[0]))
    S_inv = np.zeros((vh.shape[0], u.shape[0]))
    np.fill_diagonal(S, s, wrap=False)
    np.fill_diagonal(S_inv, 1 / s, wrap=False)

    return vh.T @ (S_inv @ (u.T @ (b @ (u @ (S_inv.T @ vh)))))


def estimate_tensor_left(X, b, arity):
    X_inv = np.linalg.pinv(X)
    result = b
    for _ in range(arity):
        result = np.tensordot(X_inv, result, axes=([1], [0]))

    return result


def quadrature_regular_grid(X, values, grid_n_samples):
    x0 = X[0]
    index = 1
    volume = 1
    n_samples = copy.copy(grid_n_samples)
    aux = n_samples[0]
    n_samples[0] = n_samples[1]
    n_samples[1] = aux
    n_samples = [1] + n_samples[:-1]
    for i in n_samples:
        index = index * i
        volume = volume * np.linalg.norm(X[index] - x0)

    return np.sum(values, axis=0) * volume


def crop_matrix(X, crop_indices):
    if len(X.shape) == 1:
        return X[crop_indices[0]]

    return np.stack([crop_matrix(f, crop_indices[1:]) for f in X[crop_indices[0]]])


# ESTIMATION OF THE RIEMANNIAN METRIC


def nearest_neighbors_indices(i, n_neighbors, distance_matrix):

    indices = np.argsort(distance_matrix[i], axis=0)[: n_neighbors + 1]
    indices = np.delete(indices, 0, axis=0)

    return indices


def get_riemannian_metric(X, Y, n_neighbors, distance_matrix):
    g = []
    dim = X.shape[1]
    for i in range(X.shape[0]):
        stop = False
        k = n_neighbors
        while not stop:
            index_neig = nearest_neighbors_indices(i, k, distance_matrix)
            X_extract = X[index_neig]
            X_vec = X_extract - X[i]
            if np.linalg.matrix_rank(X_vec) >= dim:
                stop = True
            else:
                k += 1

        Y_extract = Y[index_neig]
        Y_vec = Y_extract - Y[i]
        b = np.matmul(Y_vec, Y_vec.T)

        g.append(estimate_inner_product(X_vec, b))
    g = np.stack(g)
    return g


def compute_knn_point_derivatives(X, Y, i, n_neighbors, distance_matrix):
    stop = False
    k = n_neighbors
    dim = X.shape[1]
    while not stop:
        index_neig = nearest_neighbors_indices(i, k, distance_matrix)
        X_extract = X[index_neig]
        X_vec = X_extract - X[i]
        if np.linalg.matrix_rank(X_vec) >= dim:
            stop = True
        else:
            k += 1

    # new_shape = (k,) + Y.shape[1:]
    # b = np.zeros(new_shape)
    Y_extract = Y[index_neig]
    Y_vec = Y_extract - Y[i]

    return estimate_tensor_left(X_vec, Y_vec, 1)


def compute_knn_derivative(X, Y, n_neighbors, distance_matrix):
    der = []
    for i in range(X.shape[0]):
        der.append(compute_knn_point_derivatives(X, Y, i, n_neighbors, distance_matrix))

    return np.stack(der)


def get_knn_derivatives(X, Y, n_neighbors, distance_matrix):
    first = compute_knn_derivative(X, Y, n_neighbors, distance_matrix)
    second = compute_knn_derivative(X, first, n_neighbors, distance_matrix)
    third = compute_knn_derivative(X, second, n_neighbors, distance_matrix)

    return first, second, third


def get_partial_derivative_indices(variables, order=2):
    """
    Gets indices for partial derivatives w.r.t each variable.
    :param variables: number of variables (number of partial derivatives per order)
    :param order: max. order of derivation.
    :return: list of length order, containing permutations of every partial derivative that can be taken for
    every variable starting from the indices of the previous order.
    """
    first_order_indices = np.eye(variables, dtype=int)
    ret = [first_order_indices]

    previous_order_indices = first_order_indices
    for _ in range(1, order):
        current_order_indices = []
        for previous_order_index in previous_order_indices:
            for variable in range(variables):
                partial_derivative_index = list(previous_order_index)
                partial_derivative_index[variable] += 1
                current_order_indices.append(partial_derivative_index)

        previous_order_indices = np.array(current_order_indices)
        ret.append(previous_order_indices)

    return ret


def interpolate_derivative_function(X, Y, order=3):
    dim = Y.shape[1]

    f = []
    for i in range(dim):
        tidy_data = np.hstack((X, Y[:, i].reshape(-1, 1)))
        interp = ndsplines.make_interp_spline_from_tidy(tidy_data, range(dim), [dim])
        f.append(interp)

    partial_derivative_indices = get_partial_derivative_indices(dim, order)

    derivatives = []
    for derivative_order in range(order):
        derivatives_list = []
        for indices in partial_derivative_indices[derivative_order]:
            derivatives_list.append(
                np.stack([f[j](X, nus=indices) for j in range(dim)])
            )

        derivatives_list = np.stack(derivatives_list)
        derivatives_list = derivatives_list.reshape(
            *tuple([dim] * (derivative_order + 2)), len(X)
        )

        derivatives_list = np.moveaxis(
            derivatives_list,
            range(derivative_order + 3),
            [(x + 1) % (derivative_order + 3) for x in range(derivative_order + 3)],
        )
        derivatives.append(derivatives_list)

    return tuple(derivatives)


def interpolate_metric_derivatives(g_sp, X, n_samples, order=2):
    n = reduce(lambda x, y: x * y, n_samples)
    d = len(n_samples)

    grid = dataset_to_grid(X, n_samples, X.shape[1])

    g = np.array([g_sp[i](grid) for i in range(d ** 2)])
    g = g.T.reshape((n, d, d))

    partial_derivative_indices = get_partial_derivative_indices(d, order=order)

    derivatives = []
    for derivative_order in range(order):
        derivatives_list = []
        for indices in partial_derivative_indices[derivative_order]:
            derivatives_list.append(
                np.stack([g_sp[i](grid, nus=indices) for i in range(d ** 2)])
            )

        derivatives_list = np.array(derivatives_list).reshape(
            *tuple([d] * (derivative_order + 3)), n
        )

        derivatives_list = np.moveaxis(
            derivatives_list,
            range(derivative_order + 4),
            [(x + 1) % (derivative_order + 4) for x in range(derivative_order + 4)],
        )
        derivatives.append(derivatives_list)

    return g, *derivatives


def interpolate_metric(X, Y, n_neighbors, distance_matrix):
    n = X.shape[0]
    d = X.shape[1]
    g = get_riemannian_metric(X, Y, n_neighbors, distance_matrix)
    inter = []
    for i in range(d):
        for j in range(d):
            tidy_data = np.hstack((X, g[:, i, j].reshape(n, 1)))
            interp = ndsplines.make_interp_spline_from_tidy(
                tidy_data,
                range(d),  # columns to use as independent variable data
                [d],  # columns to use as dependent variable data
            )
            inter.append(interp)

    return inter


# TOP FUNCTIONS


def combine_derivatives(df, ddf, dddf):
    g = np.einsum("ijs,iks->ijk", df, df, optimize="greedy")
    dg = np.einsum("ijks,ils->ijkl", ddf, df) + np.einsum(
        "iks,ijls->ijkl", df, ddf, optimize="greedy"
    )
    ddg = (
        np.einsum("ijkls,ims->ijklm", dddf, df, optimize="greedy")
        + np.einsum("ikls,ijms->ijklm", ddf, ddf, optimize="greedy")
        + np.einsum("ijls,ikms->ijklm", ddf, ddf, optimize="greedy")
        + np.einsum("ils,ijkms->ijklm", df, dddf, optimize="greedy")
    )
    return g, dg, ddg


def compute_distance_matrix(X, distance_metric=None):
    if distance_metric is None:

        def distance_metric(x, y):
            return np.linalg.norm(x - y)

    return cdist(X, X, metric=distance_metric)


def estimate_sectional_curvature(
    X, Y, n_neighbors, grid_n_samples, metric_estimation="KNN"
):

    if metric_estimation == "interpolate_metric":
        distance_matrix = compute_distance_matrix(X)
        g_sp = interpolate_metric(X, Y, n_neighbors, distance_matrix)
        g, dg, ddg = interpolate_metric_derivatives(g_sp, X, grid_n_samples)
    elif metric_estimation == "KNN":
        distance_matrix = compute_distance_matrix(X)
        df, ddf, dddf = get_knn_derivatives(X, Y, n_neighbors, distance_matrix)
        g, dg, ddg = combine_derivatives(df, ddf, dddf)
    elif metric_estimation == "interpolate":
        df, ddf, dddf = interpolate_derivative_function(X, Y)
        g, dg, ddg = combine_derivatives(df, ddf, dddf)
    else:
        raise ValueError(
            f'Accepted values for param metric_estimation are {{"KNN", "interpolate", "interpolate_metric"}} ('
            f"received {metric_estimation})"
        )

    R = []
    for i in range(len(X)):
        R.append(compute_sectional_curvatures(g[i], dg[i], ddg[i]))
    return np.stack(R)


def compute_grid_norm(X, R_sq, grid_n_samples, offset, verbose=False):
    dim = X.shape[1]

    reduction_indices = [
        range(int(grid_n_samples[i] * offset), int(grid_n_samples[i] * (1 - offset)))
        for i in range(len(grid_n_samples))
    ]

    X_grid = dataset_to_grid(X, grid_n_samples, dim)
    X_red_indices = reduction_indices + [range(dim)]
    X_crop_grid = crop_matrix(X_grid, X_red_indices)

    R_sq_grid = dataset_to_grid(R_sq, grid_n_samples, R_sq.shape[1])
    R_sq_red_indices = reduction_indices + [range(R_sq.shape[1])]
    R_sq_crop_grid = crop_matrix(R_sq_grid, R_sq_red_indices)

    n_crop = 1
    grid_n_samples_crop = [len(r) for r in reduction_indices]
    for r in grid_n_samples_crop:
        n_crop = n_crop * r
    X_crop = X_crop_grid.reshape(n_crop, X.shape[1])
    R_sq_crop = R_sq_crop_grid.reshape(n_crop, R_sq.shape[1])

    if verbose:
        print("Initiating grid integration")

    return np.sqrt(
        quadrature_regular_grid(X_crop, R_sq_crop, grid_n_samples_crop).sum()
    )


def compute_adaptive_norm(X, R_sq, offset, verbose):
    dim = X.shape[1]
    result = 0
    for i in range(R_sq.shape[1]):
        tidy_data = np.hstack((X, R_sq[:, i].reshape(X.shape[0], 1)))
        interp = ndsplines.make_interp_spline_from_tidy(
            tidy_data,
            range(dim),  # columns to use as independent variable data
            [dim],  # columns to use as dependent variable data
        )

        def f(*argv):
            return interp(argv)[0, 0]

        if verbose:
            print("Initiating adaptive integration")
        result += nquad(f, [[offset, 1 - offset]] * dim, full_output=False)[0]
    return np.sqrt(result)


def L2_norm_sectional_curvature(
    X,
    Y,
    metric_estimation="KNN",
    n_neighbors=20,
    integration="grid",
    grid_n_samples=None,
    offset=0.25,
    verbose=False,
):
    """
    Available metric_estimation:
        * 'KNN': Computes metric derivatives using n_neighbors nearest neighbors.
        * 'interpolate': Interpolates the underlying function and uses it to compute the metric.
        * 'interpolate_metric': Computes the metric from the data and interpolate it to compute derivatives.

    Available integrations:
        * 'grid': Computes the integral using a grid.
        * 'adaptive': Computes the integral using an adaptive method.
    """

    if integration not in {"adaptive", "grid"}:
        raise ValueError(
            f'Accepted values for param integration are {{"adaptive", "grid"}} (received {integration})'
        )

    if metric_estimation not in {"KNN", "interpolate", "interpolate_metric"}:
        raise ValueError(
            f'Accepted values for param metric_estimation are {{"KNN", "interpolate", '
            f'"interpolate_metric"}} (received {metric_estimation})'
        )

    if integration == "grid" and grid_n_samples is None:
        X, Y, grid_n_samples = grid_data(X, Y)

    if verbose:
        print("Starting: estimation of sectional curvature")
    R_sq = (
        estimate_sectional_curvature(
            X, Y, n_neighbors, grid_n_samples, metric_estimation
        )
        ** 2
    )
    if verbose:
        print("Estimation of sectional curvature: DONE")

    if integration == "adaptive":
        return compute_adaptive_norm(X, R_sq, offset, verbose)
    return compute_grid_norm(X, R_sq, grid_n_samples, offset, verbose)


if __name__ == "__main__":

    def eval_pair(y_true, y_pred):
        os.system(
            "taskset -cp 0-%d %s > /dev/null 2>&1"
            % (multiprocessing.cpu_count(), os.getpid())
        )
        return L2_norm_sectional_curvature(
            y_true,
            y_pred,
            metric_estimation="interpolate_metric",
            verbose=False,
        )

    with open("results/projection_dataset_names.pickle", "rb") as file:
        projection_names = pickle.load(file)
    projection_results = np.load("results/projection_results.npy")

    with parallel_backend("loky"):
        print("Evaluating")
        start_time = datetime.datetime.now()
        print(f"Start time: {start_time}")
        curvatures = Parallel(n_jobs=-1)(
            delayed(eval_pair)(x, y)
            for x, y in zip([GRID] * len(projection_results), projection_results)
        )

        results = [
            (*name.split(" "), curvature)
            for name, curvature in zip(projection_names, curvatures)
        ]

        dataframe = pd.DataFrame(
            results, columns=["Instance", "Model", "Score"], index=None
        )

        end_time = datetime.datetime.now()
        print(f"End time: {end_time}")
        print(f"Time elapsed: {end_time - start_time}")

        try:
            previous_df = pd.read_csv("results.csv", index_col=0)
            if not previous_df.equals(dataframe):
                print("Different files")
                dataframe.to_csv("results2.csv")
            else:
                print("Files are identical")
        except FileNotFoundError:
            dataframe.to_csv("results.csv")
