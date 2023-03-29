from integrator import integral_solver
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """
    Gaussian integral (-inf, inf) = sqrt(pi)
    """
    return np.exp(-x**2)

integrate = integral_solver(f, float("-inf"), float("inf"), 1000)
integral_numerical = integrate.riemann()
integral_analytical = np.sqrt(np.pi)
integrate.plot(a = -2, b = 2)
print(integral_numerical)
print(integral_analytical)
