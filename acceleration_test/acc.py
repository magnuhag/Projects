import numpy as np
import matplotlib.pyplot as plt

def F_d(v):
    rho = 1.225
    C_d = 1.5
    A = 1
    return 1/2*rho*v**2*C_d*A

def F_konst(k):
    return k

n = 100000

a = np.zeros(n)
v = a.copy()
x = a.copy()
t = np.linspace(0,10, n)
dt = t[1]-t[0]
m = 2
force_val = 10

for i in range(len(a)-1):
    if i< n//2:
        a[i] = (F_konst(force_val*2)-F_d(v[i]))/m
        v[i+1] = v[i]+a[i]*dt
        x[i+1] = x[i]+v[i+1]*dt
    else:
        a[i] = (F_konst(force_val)-F_d(v[i]))/m
        v[i+1] = v[i]+a[i]*dt
        x[i+1] = x[i]+v[i+1]*dt

plt.plot(t, x, label = "early boost")

a = np.zeros(n)
v = a.copy()
x = a.copy()
t = np.linspace(0,10, n)
dt = t[1]-t[0]
m = 2
force_val = 10

for i in range(len(a)-1):
    if i > n//2:
        a[i] = (F_konst(force_val*2)-F_d(v[i]))/m
        v[i+1] = v[i]+a[i]*dt
        x[i+1] = x[i]+v[i+1]*dt
    else:
        a[i] = (F_konst(force_val)-F_d(v[i]))/m
        v[i+1] = v[i]+a[i]*dt
        x[i+1] = x[i]+v[i+1]*dt

plt.plot(t,x, label = "late boost")
plt.legend()
plt.show()
