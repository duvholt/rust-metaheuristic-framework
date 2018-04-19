import numpy as np
from math import floor, pow, sin, sqrt

def ackley(X, Y):
    x = np.array([X, Y])
    cost = np.zeros(x.shape[1:])
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    cost = -a * np.exp(-b * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(
        0.5 * (np.cos(c * x[0]) + np.cos(c * x[1]))) + a + np.exp(1.0)
    return cost


def zakharov(X, Y):
    x = np.array([X, Y])
    c = np.zeros(x.shape[1:])
    # Calculate Cost
    c = np.sum([x[i]**2 for i in range(0, 2)], axis=0) + \
        np.sum([0.5 * (i + 1) * x[i] for i in range(0, 2)], axis=0)**2 + \
        np.sum([0.5 * (i + 1) * x[i] for i in range(0, 2)], axis=0)**4
    # Return Cost
    return c


def rosenbrock(X, Y):
    a = 1.0
    b = 100.0
    return (a - X)**2 + b * (Y-X*X)**2


def himmelblau(X, Y):
    return (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2


def axis_parallel_hyper_ellipsoid_function(X, Y):
    return X**2 + 2 * Y**2


def moved_axis_parallel_hyper_ellipsoid_function(X, Y):
    return (X - 5)**2 + (2 * (Y - 5*2))**2


def sphere(X, Y):
    return X**2 + Y**2


def rastrigin(X, Y):
    a = 10.0
    return a * 2 + (X**2 - a * np.cos(2 * np.pi * X)) + (Y**2 - a * np.cos(2 * np.pi * Y))


def katsuura(X, Y):
    nx = 2
    x = [X, Y]
    tmp3 = pow(1.0 * nx, 1.2)
    f = 1.0
    for i in range(nx):
        temp = 0.0
        for j in range(1, 33):
            tmp1 = 2.0 ** j
            tmp2 = tmp1 * x[i]
            temp += np.abs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
        f *= (1.0 + (i + 1.0) * temp) ** (10.0 / tmp3)
    tmp1 = 10.0 / nx / nx
    return f * tmp1 - tmp1


def high_elliptic(X, Y):
    d = 2
    return X ** 2 + (10.0 ** 6) ** (1 / (d - 1)) * Y ** 2


def bent_cigar(X, Y):
    return X ** 2 + (10 ** 6) * Y**2


def griewank(X, Y):
    return 1.0 + 1 / 4000 * (X**2 + Y**2) - np.cos(X / np.sqrt(1)) * \
        np.cos(Y / np.sqrt(2))


def schwefel_helper(x, z):
    z_i = x + z
    d = 2
    if z_i < -500:
        return (abs(z_i) % 500.0 - 500.0) * sin(sqrt(abs((abs(z_i) % 500.0 - 500.0))))
        - (z_i + 500.0) ** 2 / (10_000.0 * d)
    elif z_i > 500.0:
        return (500.0 - (z_i % 500.0)) * sin(sqrt(abs(500.0 - z_i % 500.0)))
        - (z_i - 500.0) ** 2 / (10_000.0 * d)
    else:
        return z_i * sin(abs(z_i) ** (1.0 / 2.0))


def schwefel(X, Y):
    z = 4.209687462275036e+002
    s = 0
    d = 2
    for x in [X, Y]:
        s += np.array([[schwefel_helper(x_ij, z) for x_ij in x_i] for x_i in x])
    return 418.9829 * d - s


def weierstrass(X, Y):
    a = 0.5
    b = 3.0
    k_max = 20
    f = 0
    for x in [X, Y]:
        sum1 = 0
        sum2 = 0
        for j in range(k_max + 1):
            sum1 += a ** j * np.cos(2*np.pi * b ** j * (x + 0.5))
            sum2 += a ** j * np.cos(2*np.pi * b ** j * 0.5)
        f += sum1
    return f - 2 * sum2
