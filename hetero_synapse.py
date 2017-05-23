
# coding: utf-8

# # Heterogeneous Synapses
# 
# In this example, we demonstrate how to build an `Ensemble` that uses different synapses per dimension in the vector space, or a different synapse per neuron.
# 
# For the most general case, **``HeteroSynapse``** is a function for use within a `Node`. It accepts some vector as input, and outputs a filtered version of this vector. The dimensionality of the output vector depends on the number of elements in the list `synapses` and the boolean value `elementwise`:
#  - If `elementwise == False`, then each synapse is applied to every input dimension, resulting in an output vector that is `len(synapses)` times larger.
#  - If `elementwise == True`, then each synapse is applied separately to a single input dimension, resulting in an output vector that is size `len(synapses)`, which must also be the same as the input dimension.
# 
# ### Neuron Example
# 
# We first sample 100 neurons and 100 synapses randomly.

# In[ ]:

import numpy as np

import nengo
import nengolib
from nengolib.stats import sphere
from nengolib.synapses import HeteroSynapse

n_neurons = 100
dt = 0.001
T = 0.1
dims_in = 2

taus = nengo.dists.Uniform(0.001, 0.1).sample(n_neurons)
synapses = [nengo.Lowpass(tau) for tau in taus]
encoders = sphere.sample(n_neurons, dims_in)


# Now we create two identical ensembles, one to hold the expected result, and one to compare this with the actual result from using `HeteroSynapse`. The former is computed via brute-force, by creating a separate connection for each synapse. The latter requires a single connection to the special node. 
# 
# When `elementwise = False`, each input dimension is effectively broadcast to all of the neurons with a different synapse per neuron. We also note that since we are connecting directly to the neurons, we must embed the encoders in the transformation.

# In[ ]:

hs = HeteroSynapse(synapses, dt)

def embed_encoders(x):
    # Reshapes the vectors to be the same dimensionality as the
    # encoders, and then takes the dot product row by row.
    # See http://stackoverflow.com/questions/26168363/ for a more
    # efficient solution.
    return np.sum(encoders * hs.from_vector(x), axis=1)

with nengolib.Network() as model:
    # Input stimulus
    stim = nengo.Node(size_in=dims_in)
    for i in range(dims_in):
        nengo.Connection(
            nengo.Node(output=nengo.processes.WhiteSignal(T, high=10)),
            stim[i], synapse=None)

    # HeteroSynapse node
    syn = nengo.Node(size_in=dims_in, output=hs)

    # For comparing results
    x = [nengo.Ensemble(n_neurons, dims_in, seed=0, encoders=encoders)
         for _ in range(2)]  # expected, actual

    # Expected
    for i, synapse in enumerate(synapses):
        t = np.zeros_like(encoders)
        t[i, :] = encoders[i, :]
        nengo.Connection(stim, x[0].neurons, transform=t, synapse=synapse)

    # Actual
    nengo.Connection(stim, syn, synapse=None)
    nengo.Connection(syn, x[1].neurons, function=embed_encoders, synapse=None)

    # Probes
    p_exp = nengo.Probe(x[0].neurons, synapse=None)
    p_act = nengo.Probe(x[1].neurons, synapse=None)

# Check correctness
sim = nengo.Simulator(model, dt=dt)
sim.run(T)

assert np.allclose(sim.data[p_act], sim.data[p_exp])


# ### Vector Example
# 
# This example applies 2 synapses to their respective dimensions in a 2D-vector. We first initialize our parameters to use 20 neurons and 2 randomly chosen synapses.

# In[ ]:

n_neurons = 20
dt = 0.0005
T = 0.1
dims_in = 2
synapses = [nengo.Alpha(0.1), nengo.Lowpass(0.005)]
assert dims_in == len(synapses)

encoders = sphere.sample(n_neurons, dims_in)


# Similar to the last example, we create two ensembles, one to obtain the expected result for verification, and another to be computed using the `HeteroSynapse` node.

# In[ ]:

with nengolib.Network() as model:
    # Input stimulus
    stim = nengo.Node(size_in=dims_in)
    for i in range(dims_in):
        nengo.Connection(
            nengo.Node(output=nengo.processes.WhiteSignal(T, high=10)),
            stim[i], synapse=None)

    # HeteroSynapse Nodes
    syn_elemwise = nengo.Node(
        size_in=dims_in, 
        output=HeteroSynapse(synapses, dt, elementwise=True))

    # For comparing results
    x = [nengo.Ensemble(n_neurons, dims_in, seed=0, encoders=encoders)
         for _ in range(2)]  # expected, actual

    # Expected
    for j, synapse in enumerate(synapses):
        nengo.Connection(stim[j], x[0][j], synapse=synapse)

    # Actual
    nengo.Connection(stim, syn_elemwise, synapse=None)
    nengo.Connection(syn_elemwise, x[1], synapse=None)

    # Probes
    p_exp = nengo.Probe(x[0], synapse=None)
    p_act_elemwise = nengo.Probe(x[1], synapse=None)

# Check correctness
sim = nengo.Simulator(model, dt=dt)
sim.run(T)

assert np.allclose(sim.data[p_act_elemwise], sim.data[p_exp])


# ### Multiple Vector Example
# 
# As a final example, to demonstrate the generality of this approach, we consider the situation where we wish to apply a number of different synapses to every dimension. For instance, with a 2D input vector, we pick 3 synapses to apply to every dimension, such that our ensemble will represent a 6D-vector (one for each dimension/synapse pair).

# In[ ]:

n_neurons = 20
dt = 0.0005
T = 0.1
dims_in = 2
synapses = [nengo.Alpha(0.1), nengo.Lowpass(0.005), nengo.Alpha(0.02)]

dims_out = len(synapses)*dims_in
encoders = sphere.sample(n_neurons, dims_out)


# We also demonstrate that this can be achieved in two different ways. The first is with `elementwise=False`, by a broadcasting similar to the first example. The second is with `elementwise=True`, by replicating each synapse to align with each dimension, and then proceeding similar to the second example.

# In[ ]:

with nengolib.Network() as model:
    # Input stimulus
    stim = nengo.Node(size_in=dims_in)
    for i in range(dims_in):
        nengo.Connection(
            nengo.Node(output=nengo.processes.WhiteSignal(T, high=10)),
            stim[i], synapse=None)

    # HeteroSynapse Nodes
    syn_dot = nengo.Node(
        size_in=dims_in, output=HeteroSynapse(synapses, dt))
    syn_elemwise = nengo.Node(
        size_in=dims_out, 
        output=HeteroSynapse(np.repeat(synapses, dims_in), dt, elementwise=True))

    # For comparing results
    x = [nengo.Ensemble(n_neurons, dims_out, seed=0, encoders=encoders)
         for _ in range(3)]  # expected, actual 1, actual 2

    # Expected
    for j, synapse in enumerate(synapses):
        nengo.Connection(stim, x[0][j*dims_in:(j+1)*dims_in], synapse=synapse)

    # Actual (method #1 = matrix multiplies)
    nengo.Connection(stim, syn_dot, synapse=None)
    nengo.Connection(syn_dot, x[1], synapse=None)

    # Actual (method #2 = elementwise)
    for j in range(len(synapses)):
        nengo.Connection(stim, syn_elemwise[j*dims_in:(j+1)*dims_in], synapse=None)
    nengo.Connection(syn_elemwise, x[2], synapse=None)

    # Probes
    p_exp = nengo.Probe(x[0], synapse=None)
    p_act_dot = nengo.Probe(x[1], synapse=None)
    p_act_elemwise = nengo.Probe(x[2], synapse=None)

# Check correctness
sim = nengo.Simulator(model, dt=dt)
sim.run(T)

assert np.allclose(sim.data[p_act_dot], sim.data[p_exp])
assert np.allclose(sim.data[p_act_elemwise], sim.data[p_exp])


# In[ ]:



