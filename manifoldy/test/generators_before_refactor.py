import scipy as sc
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group

# Not part of the original, but needed for reproducibility and for tests.
from manifoldy.definitions import RANDOM_SEED


def prod_complex(z1, z2):
    return z1[0] * z2[0] - z1[1] * z2[1], z1[0] * z2[1] + z1[1] * z2[0]


def my_nintegration(f, a, b):
    return sc.integrate.quad(f, a, b)[0]


def create_curve(k, x0, v0, t_min=-1, t_max=1, t_step=0.01):
    t_num = (t_max - t_min) / t_step
    xis = np.linspace(t_min, t_max, 10 * int(t_num))
    theta_fun = np.vectorize(lambda x: my_nintegration(k, 0, x))
    theta_val = theta_fun(xis)
    sp_theta = CubicSpline(xis, theta_val)

    def _tang(xi):
        thet = sp_theta(xi)
        return prod_complex(v0, [np.cos(thet), np.sin(thet)])

    def _tang_x(xi):
        return _tang(xi)[0]

    def _tang_y(xi):
        return _tang(xi)[1]

    si = np.linspace(t_min, t_max, 10 * int(t_num))
    res_x_fun = np.vectorize(lambda x: my_nintegration(_tang_x, 0, x) + x0[0])
    res_y_fun = np.vectorize(lambda x: my_nintegration(_tang_y, 0, x) + x0[1])
    res_x_val = res_x_fun(si)
    res_y_val = res_y_fun(si)

    sp_res_x = CubicSpline(si, res_x_val)
    sp_res_y = CubicSpline(si, res_y_val)

    def _gamma(s):
        return np.array([sp_res_x(s), sp_res_y(s)])

    return _gamma


def create_spatial_curve(k, x0, v0, n, i, t_min=-1, t_max=1, t_step=0.01):
    gamma = create_curve(k, x0, v0, t_min, t_max, t_step)

    def _gamma_sp(s):
        return np.array([0] * (i) + list(gamma(s)) + [0] * (n - i - 2))

    return _gamma_sp


def create_submanifold(k, x0, v0, n, t_min=-1, t_max=1, t_step=0.01):
    d = len(k)

    if d > n - 1:
        print("Too big submanifold")

    if len(x0) != d or len(v0) != d:
        print("Dimensions do not match!")

    gammas = []
    for i in range(d):
        gammas.append(
            create_spatial_curve(k[i], x0[i], v0[i], n, i, t_min, t_max, t_step)
        )

    def _X(arg):
        suma = 0
        for i in range(d):
            suma += gammas[i](arg[i])
        return suma

    return _X


def relu(t, slope):
    if t < 0:
        return 0
    return slope * t


def get_curvature_type(curvature_type, args):
    if curvature_type == "flat":
        k = lambda t: 0
    elif curvature_type == "circle":
        k = lambda t: 2 * np.pi * (args)
    elif curvature_type == "polynomial_roll":
        k = lambda t: 4 * args * (t + 1) ** (2 * args)
    elif curvature_type == "roll":
        k = lambda t: np.exp(4 * args * t)
    elif curvature_type == "gaussian":
        k = lambda t: 1 / (0.1 * args) * np.exp(-((t) ** (2)) / (0.1 * args) ** 2)
    elif curvature_type == "sine":
        k = lambda t: (5 + (args - 1) * 10) * np.sin(2 * np.pi * t)
    elif curvature_type == "logistic":
        k = lambda t: (10 * args) / (1 + np.exp(-0.5 * t))
    elif curvature_type == "relu":
        k = lambda t: relu(t, 10 * args)

    return k


def create_curve_type(curvature_type, *args):
    return create_curve(get_curvature_type(curvature_type, *args), (0, 0), (1, 0))


def create_submanifold_type(curvature_types, args, n):
    d = len(curvature_types)
    k = [get_curvature_type(curvature_types[i], args[i]) for i in range(d)]
    if any(ks is None for ks in k):
        return None

    x0 = [(0, 0)] * d
    v0 = [(1, 1)] * d

    return create_submanifold(k, x0, v0, n)


def create_dataset_anisotropic(curvature_types, args, n, cov):
    X = create_submanifold_type(curvature_types, args, n)
    R = special_ortho_group.rvs(n, random_state=RANDOM_SEED)
    if X is None:
        print("Error during creation of the submanifold")

    def _phi(*args):
        return np.dot(
            R,
            X(*args)
            + multivariate_normal.rvs(
                mean=[0] * n, cov=cov, random_state=RANDOM_SEED
            ).T,
        ).T

    return _phi


def create_dataset(curvature_types, args, n, std):
    return create_dataset_anisotropic(
        curvature_types, args, n, np.identity(n) * std ** 2
    )
