from nengolib.synapses import PadeDelay

# Delay 15 Hz band-limited white noise by 100 ms using various orders of
# approximations:

from nengolib.signal import z
from nengo.processes import WhiteSignal
import matplotlib.pyplot as plt
process = WhiteSignal(10., high=15, y0=0)
u = process.run_steps(500)
t = process.ntrange(len(u))
plt.plot(t, (z**-100).filt(u), linestyle='--', lw=4, label="Ideal")
for order in list(range(4, 9)):
    sys = PadeDelay(.1, order=order)
    assert len(sys) == order
    plt.plot(t, sys.filt(u), label="order=%s" % order)
plt.xlabel("Time (s)")
plt.legend()
plt.show()
