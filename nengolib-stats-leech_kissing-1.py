from nengolib.stats import leech_kissing
pts = leech_kissing()

# We can visualize some of the lattice structure by projections into two
# dimensions. This scatter plot will look the same regardless of which
# two coordinates are chosen.

import matplotlib.pyplot as plt
plt.scatter(pts[:, 0], pts[:, 1], s=50)
plt.show()
