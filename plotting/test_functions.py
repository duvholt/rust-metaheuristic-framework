import numpy as np
from math import floor, pow

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
