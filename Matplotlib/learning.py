import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2 * x + 1
y2 = x**2

plt.figure()
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')

plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am X')
plt.ylabel('I am Y')

new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)

plt.show()
