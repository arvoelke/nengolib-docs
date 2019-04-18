# See :doc:`notebooks/examples/rolling_window` for a notebook example.

from nengolib.networks import RollingWindow, t_default

# Approximate the maximum of a window of width 50 ms, as well as a sampling
# of the window itself. The :class:`.Hankel` realizer happens to be better
# than the default of :class:`.Balanced` for computing the ``max`` function.

import nengo
from nengolib import Network
from nengolib.signal import Hankel
with Network() as model:
    process = nengo.processes.WhiteSignal(100., high=25, y0=0)
    stim = nengo.Node(output=process)
    rw = RollingWindow(theta=.05, n_neurons=2500, process=process,
                       neuron_type=nengo.LIFRate(), legendre=True)
    nengo.Connection(stim, rw.input, synapse=None)
    p_stim = nengo.Probe(stim)
    p_delay = nengo.Probe(rw.output)
    p_max = nengo.Probe(rw.add_output(function=np.max))
    p_window = nengo.Probe(rw.add_output(function=lambda w: w[::20]))
with nengo.Simulator(model, seed=0) as sim:
    sim.run(.5)

import matplotlib.pyplot as plt
plt.subplot(211)
plt.plot(sim.trange(), sim.data[p_stim], label="Input")
plt.plot(sim.trange(), sim.data[p_delay], label="Delay")
plt.legend()
plt.subplot(212)
plt.plot(sim.trange(), sim.data[p_window], alpha=.2)
plt.plot(sim.trange(), sim.data[p_max], c='purple', label="max(w)")
plt.legend()
plt.xlabel("Time (s)")
plt.show()

# Visualizing the canonical basis functions. The state of the
# :func:`LegendreDelay` system takes a linear combination of these
# shifted Legendre polynomials to represent the current window of history:

plt.title("canonical_basis()")
plt.plot(t_default, rw.canonical_basis())
plt.xlabel("Normalized Time (Unitless)")
plt.show()

# Visualizing the realized basis functions. This is a linear transformation
# of the above basis functions according to the realized state-space
# (see ``realizer`` parameter). The state of the **network** takes a linear
# combination of these to represent the current window of history:

plt.title("basis()")
plt.plot(t_default, rw.basis())
plt.xlabel("Normalized Time (Unitless)")
plt.show()

# Visualizing the inverse basis functions. The functions that can be
# accurately decoded are expressed in terms of the dot-product of the window
# with these functions (see :func:`.add_output` for mathematical details).

plt.title("inverse_basis().T")
plt.plot(t_default, rw.inverse_basis().T)
plt.xlabel("Normalized Time (Unitless)")
plt.show()
