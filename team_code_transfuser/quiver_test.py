import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 21, 1)
Y = np.arange(21, 0, -1)
U = np.ones((21, 21))
V = np.ones((21, 21))
V[0:10:1, ::] = -1
V[0:5:1, ::] = -4

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q, X=0.3, Y=1.1, U=10,
             label='Quiver key, length = 10', labelpos='E')

plt.show()