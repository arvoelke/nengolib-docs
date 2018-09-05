from nengolib.networks import LinearNetwork
from nengolib.synapses import Bandpass

# Implementing a 5 Hz :func:`.Bandpass` filter (i.e., a decaying 2D
# oscillator) using 1000 spiking LIF neurons:

import nengo
from nengolib import Network
from nengolib.signal import Balanced
with Network() as model:
    stim = nengo.Node(output=lambda t: 100*int(t < .01))
    sys = LinearNetwork(sys=Bandpass(freq=5, Q=10),
                        n_neurons_per_ensemble=500,
                        synapse=.1, dt=1e-3, realizer=Balanced())
    nengo.Connection(stim, sys.input, synapse=None)
    p = nengo.Probe(sys.state.output, synapse=.01)
with nengo.Simulator(model, dt=sys.dt) as sim:
    sim.run(1.)

# Note there are exactly 5 oscillations within 1 second, in response to a
# saturating impulse:

import matplotlib.pyplot as plt
plt.plot(*sim.data[p].T)
plt.xlabel("$x_1(t)$")
plt.ylabel("$x_2(t)$")
plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()
