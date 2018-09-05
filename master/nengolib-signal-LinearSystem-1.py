# A simple continuous-time integrator:

from nengolib.signal import s
integrator = 1/s
assert integrator == ~s == s**(-1)
t = integrator.trange(2.)
step = np.ones_like(t)
cosine = np.cos(t)

import matplotlib.pyplot as plt
plt.subplot(211)
plt.title("Integrating a Step Function")
plt.plot(t, step, label="Step Input")
plt.plot(t, integrator.filt(step), label="Ramping Output")
plt.legend(loc='lower center')
plt.subplot(212)
plt.title("Integrating a Cosine Wave")
plt.plot(t, cosine, label="Cosine Input")
plt.plot(t, integrator.filt(cosine), label="Sine Output")
plt.xlabel("Time (s)")
plt.legend(loc='lower center')
plt.show()

# Building up higher-order continuous systems:

sys1 = 1000/(s**2 + 2*s + 1000)   # Bandpass filtering
sys2 = 500/(s**2 + s + 500)       # Bandpass filtering
sys3 = .5*sys1 + .5*sys2          # Mixture of two bandpass
assert len(sys1) == 2  # sys1.order_den
assert len(sys2) == 2  # sys2.order_den
assert len(sys3) == 4  # sys3.order_den

plt.subplot(311)
plt.title("sys1.impulse")
plt.plot(t, sys1.impulse(len(t)), label="sys1")
plt.subplot(312)
plt.title("sys2.impulse")
plt.plot(t, sys2.impulse(len(t)), label="sys2")
plt.subplot(313)
plt.title("sys3.impulse")
plt.plot(t, sys3.impulse(len(t)), label="sys3")
plt.xlabel("Time (s)")
plt.show()

# Plotting a linear transformation of the state-space from sys3.impulse:

from nengolib.signal import balance
plt.title("balance(sys3).X.impulse")
plt.plot(t, balance(sys3).X.impulse(len(t)))
plt.xlabel("Time (s)")
plt.show()

# A discrete trajectory:

from nengolib.signal import z
trajectory = 1 - .5/z + 2/z**3 + .5/z**4
t = np.arange(7)
y = trajectory.impulse(len(t))

plt.title("trajectory.impulse")
plt.step(t, y, where='post')
plt.fill_between(t, np.zeros_like(y), y, step='post', alpha=.3)
plt.xticks(t)
plt.xlabel("Step")
plt.show()
