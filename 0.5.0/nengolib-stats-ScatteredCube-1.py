from nengolib.stats import ScatteredCube
s1 = ScatteredCube([-1, -1, -1], [1, 1, 0]).sample(1000, 3)
s2 = ScatteredCube(0, 1).sample(1000, 3)
s3 = ScatteredCube([-1, .5, 0], [-.5, 1, .5]).sample(1000, 3)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection='3d')
ax.scatter(*s1.T)
ax.scatter(*s2.T)
ax.scatter(*s3.T)
plt.show()
