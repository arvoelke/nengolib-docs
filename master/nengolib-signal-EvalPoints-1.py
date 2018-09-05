from nengolib.signal import EvalPoints

# Sampling from the state-space of an alpha synapse given band-limited
# white noise:

from nengolib import Alpha
from nengo.processes import WhiteSignal
eval_points = EvalPoints(Alpha(.5).X, WhiteSignal(10, high=20))

import matplotlib.pyplot as plt
from seaborn import jointplot
jointplot(*eval_points.sample(1000, 2).T, kind='kde')
plt.show()
