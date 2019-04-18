from nengolib.stats import random_orthogonal, sphere
rng = np.random.RandomState(seed=0)
u = sphere.sample(1000, 3, rng=rng)
u[:, 0] = 0
v = u.dot(random_orthogonal(3, rng=rng))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(111, projection='3d')
ax.scatter(*u.T, alpha=.5, label="u")
ax.scatter(*v.T, alpha=.5, label="v")
ax.patch.set_facecolor('white')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
plt.legend()
plt.show()
