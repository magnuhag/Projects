import numpy as np
import matplotlib.pyplot as plt

from integrator import IntegralSolver

#Function we want to integrate.
def function(x):
    return np.exp(-x**2)

def main():

    #Analytic solution to integral
    analyticSolution = np.sqrt(np.pi)

    integral = IntegralSolver(function, float("-inf"), float("inf"), 1000)
    integralVal = integral.riemann()

    print(f"Numeric solution = {integralVal:.4f}\
          Analytic solution = {analyticSolution:.4f}")

    error = np.abs(integralVal-analyticSolution)
    print(f"the error ε = {error:.4g}")

    t = integral.t
    transFunc = integral.integrand

    plt.plot(t, transFunc)
    plt.xlabel("t")
    plt.ylabel("g(x)")
    plt.title("Plot of transformed function")
    plt.show()
  
if __name__ =="__main__":
    main()
    