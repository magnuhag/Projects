from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class integral_solver:
    f: type(lambda x:x)
    a: float
    b: float
    n: int
    delta: float = 10**(-10)

    def func_eval(self):

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

        self.t = t
        self.integrand = evaluated_function
        self.dt = self.t[1]-self.t[0]

    def riemann(self):
        self.func_eval()
        integral = np.sum(self.integrand)*self.dt
        return integral

    def plot(self, a = -2, b = 2):

        t2 = np.linspace(a,b,len(self.t))
        plt.plot(t2, self.f(t2), label = "original f(x)")
        plt.plot(self.t, self.integrand, label = "transformed f(x)")
        plt.title("Plot of original and transformed function")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()

