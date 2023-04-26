from integrator import integral_solver
import numpy as np

def integrand(x):
    return np.exp(-x**2)

integral = integral_solver(integrand, a = -float("inf"), b = float("inf"), n = 1000)