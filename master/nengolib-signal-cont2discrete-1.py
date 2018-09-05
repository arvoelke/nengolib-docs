# Simulating an alpha synapse with a pure transmission delay:

from nengolib.signal import z, cont2discrete
from nengolib import Alpha
sys = Alpha(0.003)
dsys = z**(-20) * cont2discrete(sys, dt=sys.default_dt)
y = dsys.impulse(50)

assert np.allclose(np.sum(y), 1, atol=1e-3)
t = dsys.ntrange(len(y))

import matplotlib.pyplot as plt
plt.step(t, y, where='post')
plt.fill_between(t, np.zeros_like(y), y, step='post', alpha=.3)
plt.xlabel("Time (s)")
plt.show()
