from nengolib.signal import state_norm
from nengolib.synapses import PadeDelay
sys = PadeDelay(.1, order=4)
dt = 1e-4
y = sys.X.impulse(2500, dt=dt)

# Comparing the analytical H2-norm of the delay state to its simulated value:

assert np.allclose(np.linalg.norm(y, axis=0) * np.sqrt(dt),
                   state_norm(sys), atol=1e-4)

import matplotlib.pyplot as plt
plt.plot(sys.ntrange(len(y), dt=dt), y)
plt.xlabel("Time (s)")
plt.show()
