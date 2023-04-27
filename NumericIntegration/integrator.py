from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class IntegralSolver:
    """
    A class to perform numerical integration. Can do
    both proper and improper integrals. To do improper
    integrals, the integration boundry approaching infinity
    must be provided like this: a = float("inf"). As of now
    this class only provides the option to do Riemann integrals
    of functions f : x->R. Will perhaps add other methods
    that can integrate functions like f: R^n -> R^m.

    Attributes
    ----------
    f (function/ callable) : Integrand to be integrated

    a (float) : Lower integration limit

    b (float) : Upper integration limit

    n (int) : Number of subdivisions in Riemann sum 

    delta (float) : Small number added to/ subtracted from integration
                    limits to avoid divide by zero errors.

    Methods
    -------
    func_eval():
        Prepares function f for integration by performing
        a change of interval, if integral is improper,
        to an interval somwwhere between (-1,1). This Wiki
        article explains the method in the "Change of interval"
        section: https://en.wikipedia.org/wiki/Gaussian_quadrature.
        Otherwise this method simply evaluates f(x).
    """
    f: callable
    a: float
    b: float
    n: int
    delta: float = 10**(-10)


    def func_eval(self):
        """ func_eval()
        Nothing really interesting happening here. Just
        evaluating the function you provided.

        Parameters
        ----------
        None
        """

        if self.a == float("-inf") and self.b == float("inf"):
            t = np.linspace(-1+self.delta, 1-self.delta, self.n)
            evaluatedFunction = self.f(t/(1-t**2))*(1+t**2)/(1-t**2)**2

        elif self.a != float("-inf") and self.b == float("inf"):
            t = np.linspace(0+self.delta, 1-self.delta, self.n)
            evaluatedFunction = self.f(self.a+t/(1-t))*1/(1-t)**2

        elif self.a == float("-inf") and self.b != float("inf"):
            t = np.linspace(0+self.delta, 1-self.delta, self.n)
            evaluatedFunction= self.f(self.b-(1-t)/t)*1/t**2

        else:
            t = np.linspace(a, b, n)
            evaluatedFunction = self.f(t)

        self.t = t
        self.integrand = evaluatedFunction
        self.dt = self.t[1]-self.t[0]
        return

@dataclass
class Riemann(IntegralSolver):
    """
    A child class to IntegralSolver. Estimates the integral using 
    a Riemann sum.

    Attributes
    ----------
    f (function/ callable) : Integrand to be integrated

    a (float) : Lower integration limit

    b (float) : Upper integration limit

    n (int) : Number of subdivisions in Riemann sum 

    delta (float) : Small number added to/ subtracted from integration
                    limits to avoid divide by zero errors.

    Methods
    -------
    None.
    """

    def compute(self):
        self.func_eval()
        integral = np.sum(self.integrand)*self.dt
        return integral