# Making a sine wave process using a lambda:

from nengolib.processes import Callable
process = Callable(lambda t: np.sin(2*np.pi*t))

import matplotlib.pyplot as plt
plt.plot(process.ntrange(1000), process.run_steps(1000))
plt.xlabel("Time (s)")
plt.show()
