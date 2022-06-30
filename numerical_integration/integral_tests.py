from integrator import integral_solver
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """
    Gaussian integral (-inf, inf) = sqrt(pi)
    """
    return np.sin(x)/x

integrate = integral_solver(f, 1, float("inf"), 1000, delta = 0.1)
integral_numerical = integrate.riemann()
integral_analytical = np.pi/2
integrate.plot(a = 1, b = 10)
print(integral_numerical)
print(integral_analytical)
