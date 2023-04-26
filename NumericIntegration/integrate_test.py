from integrator import Riemann
import numpy as np

def integrand(x):
    """
    Gaussian integral. Evaluates to \sqrt(pi) 
    """
    return np.exp(-x**2)

analyticSolution = np.sqrt(np.pi)

integralVal = Riemann(integrand, float("-inf"), float("inf"), 1000).compute()

print("Numerical solution = %.4g. Analytic solution = %.4g" %(integralVal, analyticSolution))

error = np.abs(integralVal-analyticSolution)


