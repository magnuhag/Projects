import numpy as np
import matplotlib.pyplot as plt

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

    def plot(self, a = -2, b = 2):
        t2 = np.linspace(a,b,len(self.t))
        plt.plot(t2, self.f(t2), label = "original f(x)")
        plt.plot(self.t, self.integrand, label = "transformed f(x)")
        plt.title("Plot of original and transformed function")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    Integrator()
