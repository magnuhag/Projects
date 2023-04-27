import numpy as np
import matplotlib.pyplot as plt

from integrator import Riemann

def main():

    #Integral we want to compute.
    f = lambda x: np.exp(-x**2)
    #Analytic solution to integral
    analyticSolution = np.sqrt(np.pi)

    integral = Riemann(f, float("-inf"), float("inf"), 1000, delta = 10**(-15))
    integralVal = integral.compute()


    print("Numeric solution = %.4g.\
          Analytic solution = %.4g" %(integralVal, analyticSolution))

    error = np.abs(integralVal-analyticSolution)
    print("The error ε = %g" %error)

    t = integral.t
    transFunc = integral.integrand

    plt.plot(t, transFunc)
    plt.xlabel("t")
    plt.ylabel("g(x)")
    plt.title("Plot of transformed function")
    plt.show()

if __name__ =="__main__":
    main()