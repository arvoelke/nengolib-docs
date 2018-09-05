from nengolib.stats import Rd
rd = Rd().sample(10000, 2)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.scatter(*rd.T, c=np.arange(len(rd)), cmap='Blues', s=7)
plt.show()
