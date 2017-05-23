from nengolib import DoubleExp
import matplotlib.pyplot as plt
tau1 = .03
taus = np.linspace(.01, .05, 5)
plt.title(r"$\tau_1=%s$" % tau1)
for tau2 in taus:
    sys = DoubleExp(tau1, tau2)
    plt.plot(sys.ntrange(100), sys.impulse(100),
             label=r"$\tau_2=%s$" % tau2)
plt.xlabel("Time (s)")
plt.legend()
plt.show()
