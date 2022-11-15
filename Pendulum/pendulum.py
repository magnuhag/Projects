import numpy as np
import matplotlib.pyplot as plt

def avstand(x, y):
    return np.sqrt((0-x)**2+(2-y)**2)

x = np.linspace(-2,2,100000)

def y(x):
    return x**2

avstand_arr = np.zeros(len(x))

i = 0

for x_val in x:
    y_val = y(x_val)
    avstand_arr[i] = avstand(x_val, y_val)
    i += 1

print(np.min(avstand_arr))
plt.plot(x, avstand_arr)
plt.xlabel("x")
plt.ylabel("avstand fra y(x)")
plt.show()
