from nengolib.stats import Sobol
sobol = Sobol().sample(10000, 2)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.figure(figsize=(6, 6))
plt.scatter(*sobol.T, c=np.arange(len(sobol)), cmap='Blues', s=7)
plt.show()
