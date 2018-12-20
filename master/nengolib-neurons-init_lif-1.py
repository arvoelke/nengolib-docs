import nengo
from nengolib import Network
from nengolib.neurons import init_lif
# >>>
with Network() as model:
     u = nengo.Node(0)
     x = nengo.Ensemble(100, 1)
     nengo.Connection(u, x)
     p_v = nengo.Probe(x.neurons, 'voltage')
# >>>
with nengo.Simulator(model, dt=1e-4) as sim:
     init_lif(sim, x)
     sim.run(0.01)
# >>>
import matplotlib.pyplot as plt
plt.title("Initialized LIF Voltage Traces")
plt.plot(1e3 * sim.trange(), sim.data[p_v])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (Unitless)")
plt.show()
