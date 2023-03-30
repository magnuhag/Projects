A Python class for numerical evaluation of improper integrals. Proper integrals are also possible to evaluate here, but that is not the point of this exercise.
This class was made as a tool by me for me to use during my physics studies at the University of Oslo, as a quick way to check my results, 
or simply solve difficult/ impossible integrals when the integral was simply "in my way," so to speak.

This class uses the following transformations to compute integrals

$$
\int\_{-\infty}^{\infty}{f(x) dx}=\int\_{-1}^1{f\Biggr(\frac{t}{1-t^2}\Biggr)\frac{1+t^2}{(1-t^2)^2}dt}
$$

$$
\int_{a}^{\infty}f(x)dx = \int_{0}^{1}f\Biggr(a+\frac{t}{1-t}\Biggr)\frac{1}{(1-t)^2}dt
$$

$$
\int_{-\infty}^{a}f(x)dx=\int_{0}^{1}f\Biggr(a-\frac{1-t}{t}\Biggr)\frac{1}{t^2}dt
$$

These transforms are fairly limited when it comes to most rational functions I've encountered
