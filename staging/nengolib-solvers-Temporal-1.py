# Below we use the temporal solver to learn a filtered communication-channel
# (the identity function) using 100 low-threshold spiking (LTS) Izhikevich
# neurons. The training and test data are sampled independently from the
# same band-limited white-noise process.

from nengolib import Temporal, Network
import nengo
neuron_type = nengo.Izhikevich(coupling=0.25)
tau = 0.005
process = nengo.processes.WhiteSignal(period=5, high=5, y0=0, rms=0.3)
eval_points = process.run_steps(5000)
with Network() as model:
    stim = nengo.Node(output=process)
    x = nengo.Ensemble(100, 1, neuron_type=neuron_type)
    out = nengo.Node(size_in=1)
    nengo.Connection(stim, x, synapse=None)
    nengo.Connection(x, out, synapse=None,
                     eval_points=eval_points,
                     function=nengo.Lowpass(tau).filt(eval_points),
                     solver=Temporal(synapse=tau))
    p_actual = nengo.Probe(out, synapse=tau)
    p_ideal = nengo.Probe(stim, synapse=tau)
with nengo.Simulator(model) as sim:
    sim.run(5)

import matplotlib.pyplot as plt
plt.plot(sim.trange(), sim.data[p_actual], label="Actual")
plt.plot(sim.trange(), sim.data[p_ideal], label="Ideal")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
