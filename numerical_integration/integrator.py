from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class integral_solver:
    """
    A class to perform numerical integration. Can do
    both proper and improper integrals. To do improper
    integrals, the integration boundry approaching infinity
    must be provided like this: a = float(inf). As of now
    this class only provides the option to do Riemann integrals
    of functions f : R->R. Will perhaps add other methods
    that can integrate functions like f: R^n -> R^m.

    Attributes
    ----------
    f : function
        Integrand to be integrated
    a : float
        lower integration limit
    b : float
        upper integration limit
    n : int
        number of subdivisions in Riemann sum
    delta: float
        small number added to/ subtracted from integration
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

    f: type(lambda x:x)
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
        #If-check to see if we're doing improper integral. If-check
        #can break if a and b are switched, for example in a = float(-inf).
        #But if that is done, integral will be wrong anyways.
        if self.a != float("-inf") and self.b != ("inf"):
            self.transformed == False
        else:
            self.transformed == True
            evaluated_function = np.zeros(self.n)

        if self.a == float("-inf") and self.b == float("inf"):
            t = np.linspace(-1+self.delta, 1-self.delta, self.n)
            evaluated_function = self.f(t/(1-t**2))*(1+t**2)/(1-t**2)**2

        elif self.a != float("-inf") and self.b == float("inf"):
            t = np.linspace(0+self.delta, 1-self.delta, self.n)
            evaluated_function = self.f(self.a+t/(1-t))*1/(1-t)**2

        elif self.a == float("-inf") and self.b != float("inf"):
            t = np.linspace(0+self.delta, 1-self.delta, self.n)
            evaluated_function= self.f(self.b-(1-t)/t)*1/t**2

        else:
            t = np.linspace(a, b, n)
            evaluated_function = self.f(t)

        self.t = t
        self.integrand = evaluated_function
        self.dt = self.t[1]-self.t[0]

    def riemann(self):
        """riemann()
        Uses Riemann sum to approximate integral

        Parameters
        ----------
        None

        Returns
        -------
        integral
        """
        self.func_eval()
        a = njksdv
        integral = np.sum(self.integrand)*self.dt
        return integral

    def plot(self, orig = False, trans = True):
        """plot(orig = False, trans = True)
        Plots either original function, transformed function, or both.
        Useful for visualizing the transformed function

        Prameters
        ---------
        orig : Bool, optional
            plot original function (non-transformed)
        trans : Bool, optional
            plot transformed function. Only applicable when integration
            bounderies are float("inf") or float("-inf")
        """

        if orig == True:
            plt.plot(self.t, self.f(self.t), label = "original f(x)")

        if self.transformed == True and trans == True:
            plt.plot(self.t, self.integrand, label = "transformed f(x)")

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()
