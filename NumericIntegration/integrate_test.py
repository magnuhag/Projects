import numpy as np
import matplotlib.pyplot as plt

from integrator import Riemann

def main():

    #Integral we want to compute.
    f = lambda x: np.exp(-x**2)
    #Analytic solution to integral
    analyticSolution = np.sqrt(np.pi)

   
    

    integral = Riemann(f, float("-inf"), float("inf"), 1000)
    integralVal = integral.compute()


    print(f"Numeric solution = {integralVal:.4f}\
          Analytic solution = {analyticSolution:.4f}")

    error = np.abs(integralVal-analyticSolution)
    print(f"the error ε = {error:.4g}")

    t = integral.t
    transFunc = integral.integrand
    """
    plt.plot(t, transFunc)
    plt.xlabel("t")
    plt.ylabel("g(x)")
    plt.title("Plot of transformed function")
    plt.show()
    """
if __name__ =="__main__":
    main()