from integrator import Integrator
import numpy as np

def f(x):
    """
    Gaussian integral (-inf, inf) = sqrt(pi)
    """
    return np.exp(-x**2)

integrate = Integrator(f, float("-inf"), float("inf"), 1000)
integral_numerical = integrate.riemann()
integral_analytical = np.sqrt(np.pi)
integrate.plot(a = -4, b = 4)
print(integral_numerical)
print(integral_analytical)
print(np.abs(integral_analytical-integral_numerical))
