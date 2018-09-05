# See :doc:`notebooks/examples/full_force_learning` for an example of how to
# use RLS to learn spiking FORCE [1]_ and "full-FORCE" networks in Nengo.

# Below, we compare :class:`nengo.PES` against :class:`.RLS`, learning a
# feed-forward communication channel (identity function), online,
# and starting with 100 spiking LIF neurons from scratch (zero weights).
# A faster learning rate for :class:`nengo.PES` results in over-fitting to
# the most recent online example, while a slower learning rate does not
# learn quickly enough. This is a general problem with greedy optimization.
# :class:`.RLS` performs better since it is L2-optimal.

from nengolib import RLS, Network
import nengo
from nengo import PES
tau = 0.005
learning_rules = (PES(learning_rate=1e-3, pre_tau=tau),
                  RLS(learning_rate=1e-5, pre_synapse=tau))

with Network() as model:
    u = nengo.Node(output=lambda t: np.sin(2*np.pi*t))
    probes = []
    for lr in learning_rules:
        e = nengo.Node(size_in=1,
                       output=lambda t, e: e if t < 1 else 0)
        x = nengo.Ensemble(100, 1, seed=0)
        y = nengo.Node(size_in=1)
# >>>
        nengo.Connection(u, e, synapse=None, transform=-1)
        nengo.Connection(u, x, synapse=None)
        conn = nengo.Connection(
            x, y, synapse=None, learning_rule_type=lr,
            function=lambda _: 0)
        nengo.Connection(y, e, synapse=None)
        nengo.Connection(e, conn.learning_rule, synapse=tau)
        probes.append(nengo.Probe(y, synapse=tau))
    probes.append(nengo.Probe(u, synapse=tau))

with nengo.Simulator(model) as sim:
    sim.run(2.0)

import matplotlib.pyplot as plt
plt.plot(sim.trange(), sim.data[probes[0]],
         label=str(learning_rules[0]))
plt.plot(sim.trange(), sim.data[probes[1]],
         label=str(learning_rules[1]))
plt.plot(sim.trange(), sim.data[probes[2]],
         label="Ideal", linestyle='--')
plt.vlines([1], -1, 1, label="Training -> Testing")
plt.ylim(-2, 2)
plt.legend(loc='upper right')
plt.xlabel("Time (s)")
plt.show()
