#!/usr/bin/env python
# coding: utf-8

import copy
import ndsplines
import numpy as np
from scipy.special import comb
from scipy import integrate
from scipy.interpolate import griddata

# COMPUTATION OF CHRISTOFFEL SYMBOLS AND CURVATURE


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
    # shap = [d] + n_samples
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
    for i, e in enumerate(s):
        S[i, i] = e
        S_inv[i, i] = 1 / e

    return vh.T @ (S_inv @ (u.T @ (b @ (u @ (S_inv.T @ vh)))))


def estimate_tensor_left(X, b, arity):
    X_inv = np.linalg.pinv(X)
    result = b
    for _ in range(arity):
        result = np.tensordot(X_inv, result, axes=([1], [0]))

    return result


def quadrature_regular_grid(X, values, grid_n_samples):
    # d = len(grid_n_samples)
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


def get_indexes_best_neighbors(X, K, i, d):
    index_neig = [-1] * K
    dist_neig = np.array([np.inf] * K)

    for j in range(X.shape[0]):
        if d(X[i], X[j]) < np.max(dist_neig) and j != i:
            index_max = np.argmax(dist_neig)
            index_neig[index_max] = j
            dist_neig[index_max] = d(X[i], X[j])

    return index_neig


def get_riemannian_metric(X, Y, K):
    d = lambda x, y: np.linalg.norm(x - y)

    g = []
    dim = X.shape[1]
    for i in range(X.shape[0]):
        stop = False
        k = K
        while not stop:
            index_neig = get_indexes_best_neighbors(X, k, i, d)
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


def compute_derivative_point(X, Y, i, K):
    d = lambda x, y: np.linalg.norm(x - y)

    stop = False
    k = K
    dim = X.shape[1]
    while not stop:
        index_neig = get_indexes_best_neighbors(X, k, i, d)
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


def compute_derivative(X, Y, K):
    der = []
    for i in range(X.shape[0]):
        der.append(compute_derivative_point(X, Y, i, K))

    return np.stack(der)


def compute_derivative_function(X, Y, K):
    df = compute_derivative(X, Y, K)
    ddf = compute_derivative(X, df, K)
    dddf = compute_derivative(X, ddf, K)

    return Y, df, ddf, dddf


def interpolate_derivative_function(X, Y):
    dim = Y.shape[1]
    n = X.shape[0]

    f = []
    for i in range(dim):
        tidy_data = np.hstack((X, Y[:, i].reshape(n, 1)))
        interp = ndsplines.make_interp_spline_from_tidy(
            tidy_data,
            range(dim),  # columns to use as independent variable data
            [dim],  # columns to use as dependent variable data
        )
        f.append(interp)

    derivs, dderivs, ddderivs = compute_derivatives_dirs(dim, 3)

    # DERIVATIVE
    df = []
    for der in derivs:
        dfj = []
        for j in range(dim):
            dfj.append(f[j](X, nus=der))
        df.append(np.stack(dfj))
    df = np.stack(df)
    df = np.moveaxis(df.reshape(dim, dim, n), [0, 1, 2], [1, 2, 0])

    # SECOND DERIVATIVES (HESSIAN)
    ddf = []
    for der in dderivs:
        ddfj = []
        for j in range(dim):
            ddfj.append(f[j](X, nus=der))
        ddf.append(np.stack(ddfj))
    ddf = np.stack(ddf)
    ddf = np.moveaxis(ddf.reshape(dim, dim, dim, n), [0, 1, 2, 3], [1, 2, 3, 0])

    # THIRD DERIVATIVES
    dddf = []
    for der in ddderivs:
        dddfj = []
        for j in range(dim):
            dddfj.append(f[j](X, nus=der))
        dddf.append(np.stack(dddfj))
    dddf = np.stack(dddf)
    dddf = np.moveaxis(
        dddf.reshape(dim, dim, dim, dim, n), [0, 1, 2, 3, 4], [1, 2, 3, 4, 0]
    )

    return Y, df, ddf, dddf


def interpolate_metric_derivatives(g_sp, X, n_samples):
    n = 1
    for n_s in n_samples:
        n = n * n_s
    d = len(n_samples)
    grid = dataset_to_grid(X, n_samples, X.shape[1])
    g = np.array([g_sp[i](grid) for i in range(d * d)])
    g = g.T.reshape((n, d, d))

    derivs, dderivs = compute_derivatives_dirs(d)

    dg = []
    for der in derivs:
        dg.append(np.array([g_sp[i](grid, nus=der) for i in range(d * d)]))

    dg = np.array(dg).reshape((d, d, d, n))
    dg = np.moveaxis(dg, [0, 1, 2, 3], [1, 2, 3, 0])

    ddg = []
    for der in dderivs:
        ddg.append(np.array([g_sp[i](grid, nus=der) for i in range(d * d)]))

    ddg = np.array(ddg).reshape((d, d, d, d, n))
    ddg = np.moveaxis(ddg, [0, 1, 2, 3, 4], [1, 2, 3, 4, 0])

    return g, dg, ddg


def interpolate_metric(X, Y, K):
    n = X.shape[0]
    d = X.shape[1]
    g = get_riemannian_metric(X, Y, K)
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


def compute_derivatives_dirs(d, n_der=2):
    derivs = []
    for i in range(d):
        der = np.array([0] * d)
        der[i] = 1
        derivs.append(der)

    if n_der == 1:
        return derivs

    dderivs = []
    for i in derivs:
        for j in range(d):
            der = copy.deepcopy(i)
            der[j] = der[j] + 1
            dderivs.append(der)

    if n_der == 2:
        return derivs, dderivs

    ddderivs = []
    for i in dderivs:
        for j in range(d):
            der = copy.deepcopy(i)
            der[j] = der[j] + 1
            ddderivs.append(der)

    return derivs, dderivs, ddderivs


# TOP FUNCTIONS


def estimate_metric_derivatives(X, Y, K, grid_n_samples, metric_estimation="KNN"):
    if metric_estimation == "KNN":
        _, df, ddf, dddf = compute_derivative_function(X, Y, K)
    elif metric_estimation == "interpolate":
        _, df, ddf, dddf = interpolate_derivative_function(X, Y)
    elif metric_estimation == "interpolate_metric":
        g_sp = interpolate_metric(X, Y, K)
        g, dg, ddg = interpolate_metric_derivatives(g_sp, X, grid_n_samples)
        return g, dg, ddg

    g = np.einsum("ijs,iks->ijk", df, df)
    dg = np.einsum("ijks,ils->ijkl", ddf, df) + np.einsum("iks,ijls->ijkl", df, ddf)
    ddg = (
        np.einsum("ijkls,ims->ijklm", dddf, df)
        + np.einsum("ikls,ijms->ijklm", ddf, ddf)
        + np.einsum("ijls,ikms->ijklm", ddf, ddf)
        + np.einsum("ils,ijkms->ijklm", df, dddf)
    )

    return g, dg, ddg


def estimate_sectional_curvature(X, Y, K, grid_n_samples, metric_estimation="KNN"):
    n = X.shape[0]

    g, dg, ddg = estimate_metric_derivatives(X, Y, K, grid_n_samples, metric_estimation)

    R = []
    for i in range(n):
        R.append(compute_sectional_curvatures(g[i], dg[i], ddg[i]))
    R = np.stack(R)

    return R


def compute_norm(
    X, R_sq, grid_n_samples, offset, integration="adaptative", verbose=False
):
    dim = X.shape[1]
    if integration == "adaptative":
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
                print("Initiating adaptative integration")
            result += integrate.nquad(
                f, [[offset, 1 - offset]] * dim, full_output=False
            )[0]
        return np.sqrt(result)

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


def L2_norm_sectional_curvature(
    X,
    Y,
    metric_estimation="KNN",
    K=20,
    integration="grid",
    grid_n_samples=None,
    offset=0.25,
    verbose=False,
):
    """
    Available metric_estimation:
        * 'KNN': Computes metric derivatives using K nearest neighbors.
        * 'interpolate': Interpolates the underlying function and uses it to compute the metric.
        * 'interpolate_metric': Computes the metric from the data and interpolate it to compute derivatives.

    Availabel integration:
        * 'grid': Computes the integral using a grid.
        * 'adaptative': Computes the integral using an adaptative method.
    """
    if grid_n_samples is None:
        X, Y, grid_n_samples = grid_data(X, Y)
    if verbose:
        print("Starting: estimation of sectional curvature")
    R = estimate_sectional_curvature(X, Y, K, grid_n_samples, metric_estimation)
    R_sq = R ** 2
    if verbose:
        print("Estimation of sectional curvature: DONE")

    return compute_norm(X, R_sq, grid_n_samples, offset, integration, verbose)
