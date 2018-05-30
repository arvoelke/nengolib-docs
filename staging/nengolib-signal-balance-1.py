# See :doc:`notebooks/research/linear_model_reduction` for a notebook
# example.

from nengolib.signal import balance, s
before = 10 / ((s + 10) * (s + 20) * (s + 30) * (s + 40))
after = balance(before)

# Effect of balancing some arbitrary system:

import matplotlib.pyplot as plt
length = 500
plt.subplot(211)
plt.title("Impulse - Before")
plt.plot(before.ntrange(length), before.X.impulse(length))
plt.subplot(212)
plt.title("Impulse - After")
plt.plot(after.ntrange(length), after.X.impulse(length))
plt.xlabel("Time (s)")
plt.show()
