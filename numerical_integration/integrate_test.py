from integrator import IntegralSolver
import numpy as np

def integrand(x):
    """
    Gaussian integral. Evaluates to \sqrt(pi) 
    """
    return np.exp(-x**2)

analyticSolution = np.sqrt(np.pi)

integral = IntegralSolver(integrand, float("-inf"), float("inf"), 1000)
integralVal = integral.riemann()
print("Numerical solution = %.4g. Analytic solution = %.4g" %(integralVal, analyticSolution))

error = np.abs(integralVal-analyticSolution)

print("Error = %g" %error)

integral.plot(orig = False, trans = False)
