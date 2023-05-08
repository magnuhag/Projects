import numpy as np

class IntegralSolver:

    def __init__(self, f: callable, a: float, b: float, n: int, 
                 delta: float = np.finfo(float).eps):
        """
        A class to perform numerical integration. Can do
        both proper and improper integrals. To do improper
        integrals, the integration boundry (or boundaries) approaching 
        infinity must be provided like this: 

        a = float("-inf")
        b = float("inf")

        As of now this class only provides the 
        option to do Riemann and Trapezoidal integrals of functions 
        of the type f : x->R. 
        Will perhaps add other methods that can integrate functions 
        like f: R^n -> R^m.

        Attributes
        ----------
        f (function/ callable) : Function to be integrated (integrand)

        a (float) : Lower integration limit

        b (float) : Upper integration limit

        n (int) : Number of subdivisions in Riemann sum 

        delta (float) : Small number added to/ subtracted from integration
                        limits to avoid divide by zero errors. This value 
                        is defaulted to machine epsilon. Perhaps not ideal.

        integrand (None) : Initialized by the constructor to be None type.
                           When calling func_eval(), integrand will be a
                           numpy array containing values of the evaluated 
                           function.

        Methods
        -------
        func_eval():
            Prepares function f for integration by performing
            a change of interval, if integral is improper,
            to an interval somewhere between (-1,1). This Wiki
            article explains the method in the "Change of interval"
            section: https://en.wikipedia.org/wiki/Gaussian_quadrature.
            Otherwise this method simply evaluates f(x). A lot of mathemtical
            detail is swept under the rug her, or is simply ignored.

        riemann():
            Evalutes the left Riemann sum.
        
        trapezoidal():
            Uses trapezoidal rule to evaluate integral.
        """
        self.f = f
        self.a = a 
        self.b = b 
        self.n = n 
        self.delta = delta
        self.integrand = None

    def func_eval(self):
        """ func_eval()
        Nothing really interesting happening here. Just
        evaluating the function you provided.

        Parameters
        ----------
        None
        """

        assert self.a < self.b, "a must be smaller than b"
            
        if abs(self.a) == self.b == float("inf"):   
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

    def riemann(self):
        """Returns value of integral as computed by Riemann sum """
        self.func_eval()
        integral = np.sum(self.integrand)*self.dt
        return integral

    def trapezoidal(self):
        """Returns value of integral as computed by trapezoidal rule"""
        self.func_eval()
        partialSum = 0
        for i in range(1, self.n):
            partialSum += (self.integrand[i-1]+self.integrand[i])
        #Factoring out this part to save FLOPs. Perhaps needless optimization
        #but we might avoid compounding round-off error 
        integral = partialSum/2*self.dt
        return integral
        
