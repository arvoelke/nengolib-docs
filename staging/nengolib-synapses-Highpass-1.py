from nengolib.synapses import Highpass

# Evaluate the highpass in the frequency domain with a time-constant of 10 ms
# and with a variety of orders:

tau = 1e-2
orders = list(range(1, 9))
freqs = np.linspace(0, 50, 100)  # to evaluate

import matplotlib.pyplot as plt
plt.title(r"$\tau=%s$" % tau)
for order in orders:
    sys = Highpass(tau, order)
    assert len(sys) == order
    plt.plot(freqs, np.abs(sys.evaluate(freqs)),
             label=r"order=%s" % order)
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()
