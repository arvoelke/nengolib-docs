# Simulate a Nengo network using a box filter of 10 ms for a synapse:

from nengolib.synapses import BoxFilter
import nengo
with nengo.Network() as model:
    stim = nengo.Node(output=lambda _: np.random.randn(1))
    p_stim = nengo.Probe(stim)
    p_box = nengo.Probe(stim, synapse=BoxFilter(10))
with nengo.Simulator(model) as sim:
    sim.run(.1)

import matplotlib.pyplot as plt
plt.step(sim.trange(), sim.data[p_stim], label="Noisy Input", alpha=.5)
plt.step(sim.trange(), sim.data[p_box], label="Box-Filtered")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
