import numpy as np


class Integrator:

    def __init__(self, f, a, b, n):

        self.f = f
        self.a = a
        self.b = b
        self.n = n
        self.integrand = 0
        self.dt = 0

    def func_eval(self):

        evaluated_function = np.zeros(self.n)

        if self.a == float("-inf") and self.b == float("inf"):
            t = np.linspace(-1, 1, self.n)
            evaluated_function[0] = 0
            evaluated_function[-1] = 0
            evaluated_function[1:-2] = self.f(t[1:-2]/(1-t[1:-2]**2))*(1+t[1:-2]**2)/(1-t[1:-2]**2)**2

        elif self.a != float("-inf") and self.b == float("inf"):
            t = np.linspace(0, 1, self.n)
            evaluated_function[0] = self.f(self.a)
            evaluated_function[-1] = 0
            evaluated_function[1:-2] = self.f(self.a+t[1:-2]/(1-t[1:-2]))*1/(1-t[1:-2])**2

        elif self.a == float("-inf") and self.b != float("inf"):
            t = np.linspace(0, 1, self.n)
            evaluated_function[0] = 0
            evaluated_function[-1] = self.f(self.b)
            evaluated_function[1:-2] = self.f(self.b-(1-t[1:-2])/t[1:-2])*1/t[1:-2]**2

        self.t = t
        self.integrand = evaluated_function
        self.dt = self.t[1]-self.t[0]

    def riemann(self):
        self.func_eval()
        integral = np.sum(self.integrand)*self.dt
        return integral

def f(x):
    """
    Gaussian integral (-inf, inf) = sqrt(pi)
    """
    return np.exp(-x**2)

integrate = Integrator(f, float("-inf"), float("inf"), 1000)
integral_numerical = integrate.riemann()
integral_analytical = np.sqrt(np.pi)
print(integral_numerical)
print(integral_analytical)
print(np.abs(integral_analytical-integral_numerical))
