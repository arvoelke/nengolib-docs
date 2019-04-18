from nengolib.synapses import pade_delay_error
abs(pade_delay_error(1, order=6))
# 0.0070350205992081461

# This means that for ``order=6`` and frequencies less than ``1/theta``,
# the approximation error is less than one percent!

# Now visualize the error across a range of frequencies, with various orders:

import matplotlib.pyplot as plt
freq_times_theta = np.linspace(0, 5, 1000)
for order in range(4, 9):
    plt.plot(freq_times_theta,
             abs(pade_delay_error(freq_times_theta, order=order)),
             label="order=%s" % order)
plt.xlabel(r"Frequency $\times \, \theta$ (Unitless)")
plt.ylabel("Absolute Error")
plt.legend()
plt.show()
