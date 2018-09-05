# See :doc:`notebooks/research/discrete_comparison` for a notebook example.

from nengolib.synapses import ss2sim, PadeDelay

# Map the state of a balanced :func:`PadeDelay` onto a lowpass synapse:

import nengo
from nengolib.signal import balance
sys = balance(PadeDelay(.05, order=6))
synapse = nengo.Lowpass(.1)
mapped = ss2sim(sys, synapse, synapse.default_dt)
assert np.allclose(sys.C, mapped.C)
assert np.allclose(sys.D, mapped.D)

# Simulate the mapped system directly (without neurons):

process = nengo.processes.WhiteSignal(1, high=10, y0=0)
with nengo.Network() as model:
    stim = nengo.Node(output=process)
    x = nengo.Node(size_in=len(sys))
    nengo.Connection(stim, x, transform=mapped.B, synapse=synapse)
    nengo.Connection(x, x, transform=mapped.A, synapse=synapse)
    p_stim = nengo.Probe(stim)
    p_actual = nengo.Probe(x)
with nengo.Simulator(model) as sim:
    sim.run(.5)

# The desired dynamics are implemented perfectly:

target = sys.X.filt(sim.data[p_stim])
assert np.allclose(target, sim.data[p_actual])

import matplotlib.pyplot as plt
plt.plot(sim.trange(), target, linestyle='--', lw=4)
plt.plot(sim.trange(), sim.data[p_actual], alpha=.5)
plt.show()
