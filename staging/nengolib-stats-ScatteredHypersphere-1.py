from nengolib.stats import ball, sphere
b = ball.sample(1000, 2)
s = sphere.sample(1000, 3)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.title("Ball")
plt.scatter(*b.T, s=10, alpha=.5)
ax = plt.subplot(122, projection='3d')
ax.set_title("Sphere").set_y(1.)
ax.patch.set_facecolor('white')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
ax.scatter(*s.T, s=10, alpha=.5)
plt.show()
