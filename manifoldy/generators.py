import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import multivariate_normal
from scipy.stats import special_ortho_group
from scipy.integrate import quad


CURVATURES = {
    "flat": lambda x: lambda t: 0,
    "circle": lambda x: lambda t: x * np.pi * 2,
    "polynomial_roll": lambda x: lambda t: x * 4 * (t + 1) ** (x * 2),
    "roll": lambda x: lambda t: np.exp(x * t * 4),
    "gaussian": lambda x: lambda t: 1 / (x * 0.1) * np.exp(-(t ** 2) / (x * 0.1) ** 2),
    "sine": lambda x: lambda t: (5 + (x - 1) * 10) * np.sin(t * np.pi * 2),
    "logistic": lambda x: lambda t: (x * 10) / (1 + np.exp(-0.5 * t)),
    "relu": lambda x: lambda t: 0 if t < 0 else x * t * 10,
}


def create_spatial_curve(k, x0, v0, n, i, t_min=-1, t_max=1, t_step=0.01):
    t_num = (t_max - t_min) // t_step
    xis = np.linspace(t_min, t_max, 10 * int(t_num))

    theta_fun = np.vectorize(lambda x: quad(k, 0, x)[0])
    theta_val = theta_fun(xis)

    sp_theta = CubicSpline(xis, theta_val)

    def _tang(xi):
        thet = sp_theta(xi)
        complex_product = complex(v0) * complex(np.cos(thet), np.sin(thet))
        return complex_product.real, complex_product.imag

    def _tang_x(xi):
        return _tang(xi)[0]

    def _tang_y(xi):
        return _tang(xi)[1]

    si = np.linspace(t_min, t_max, 10 * int(t_num))

    res_x_fun = np.vectorize(lambda x: quad(_tang_x, 0, x)[0] + x0[0])
    res_y_fun = np.vectorize(lambda x: quad(_tang_y, 0, x)[0] + x0[1])

    res_x_val = res_x_fun(si)
    res_y_val = res_y_fun(si)

    sp_res_x = CubicSpline(si, res_x_val)
    sp_res_y = CubicSpline(si, res_y_val)

    def _gamma_sp(s):
        return np.array(
            [0] * i + list(np.array([sp_res_x(s), sp_res_y(s)])) + [0] * (n - i - 2)
        )

    return _gamma_sp


def create_submanifold_type(curvature_types, args, n, t_min=-1, t_max=1, t_step=0.01):
    k = [CURVATURES[curvature_types[i]](args[i]) for i in range(len(curvature_types))]

    if any(ks is None for ks in k):
        raise ValueError(
            f"Some curvature yielded an error: {list(filter(lambda curve: curve is None, k))[0]}"
        )

    x0 = [(0, 0)] * len(curvature_types)
    v0 = [(1, 1)] * len(curvature_types)

    gammas = []
    for i, items in enumerate(zip(k, x0, v0)):
        curvature, x, v = items
        gammas.append(create_spatial_curve(curvature, x, v, n, i, t_min, t_max, t_step))

    def _X(arg):
        return sum([gammas[i](arg[i]) for i in range(len(k))])

    return _X


def create_dataset(curvature_types, args, n, std):

    X = create_submanifold_type(curvature_types, args, n)
    R = special_ortho_group.rvs(n)
    cov = np.identity(n) * std ** 2

    if len(curvature_types) > n - 1:
        raise ValueError("Too big submanifold")

    def _phi(x):
        return np.dot(R, X(x) + multivariate_normal.rvs(mean=[0] * n, cov=cov).T).T

    return _phi
