# Simulate a Nengo network using a discrete delay of half a second for a
# synapse:

from nengolib.synapses import DiscreteDelay
import nengo
with nengo.Network() as model:
    stim = nengo.Node(output=lambda t: np.sin(2*np.pi*t))
    p_stim = nengo.Probe(stim)
    p_delay = nengo.Probe(stim, synapse=DiscreteDelay(500))
with nengo.Simulator(model) as sim:
    sim.run(1.)

import matplotlib.pyplot as plt
plt.plot(sim.trange(), sim.data[p_stim], label="Stimulus")
plt.plot(sim.trange(), sim.data[p_delay], label="Delayed")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
