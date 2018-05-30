from nengolib import Lowpass
import matplotlib.pyplot as plt
taus = np.linspace(.01, .05, 5)
for tau in taus:
    sys = Lowpass(tau)
    plt.plot(sys.ntrange(100), sys.impulse(100),
             label=r"$\tau=%s$" % tau)
plt.xlabel("Time (s)")
plt.legend()
plt.show()
