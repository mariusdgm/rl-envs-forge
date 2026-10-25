# NetworkGraph

NetworkGraph models how opinions spread through a network. At each step, you choose
how much control to apply to each node. Control moves opinions toward a target,
then the network evolves according to the selected dynamics model.

## Getting started

```python
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph

env = NetworkGraph(num_agents=10, max_u=0.1, seed=0)
opinions, info = env.reset(seed=0)
env.action_space.seed(0)

for _ in range(100):
    action = env.action_space.sample()
    opinions, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

The observation is a vector of `num_agents` opinions. The action is a vector of
the same length, with each entry between zero and that node's `max_u`. You can
set `max_u` and `desired_opinion` per node by passing arrays instead of scalars.
`control_resistance` takes an array or `None`.
Observations use `float64` and actions use `float32`. Observations returned by
`reset()` and `step()` are independent copies; changing them does not change
the simulation. Use `env.state` or `env.opinions` to inspect or explicitly edit
the live state.

## Choosing a network

By default, the environment generates a random graph. Use
`graph_connection_distribution="uniform"`, `"normal"`, or `"exponential"` to
choose how connection probabilities are sampled. `graph_connection_params`
accepts `mean` and `std` for the normal distribution, or `scale` for the
exponential distribution. Set `bidirectional_prob=1.0` for an undirected graph.
Random generation repairs nodes with no incident edges. In a directed graph,
this counts incoming and outgoing edges together; it does not guarantee an
outgoing edge, weak connectivity, or strong connectivity. Random graphs retain
binary edges even when `use_weighted_edges=True`.
Random generation requires at least two nodes; supply a matrix explicitly for
a one-node model.

For a Barabasi-Albert network, use `graph_model="barabasi_albert"`. `ba_m` sets
the number of attachments for each new node, and `ba_prune_max_frac` controls
how many directed edges may be removed. For at least two nodes, `ba_m` must
satisfy `1 <= ba_m < num_agents`, and the pruning fraction must be in `[0,1]`.
The one-node BA path returns a zero matrix, which is incompatible with
FJ. The resulting influence matrix is
normalized by row, leaving rows with no outgoing edges at zero. This
normalization is part of the Barabasi-Albert generator; it is not
applied to supplied matrices or random graphs. `ba_connectivity` selects the
acceptance rule:

| Criterion | Meaning on the directed support of the adjacency matrix |
| --- | --- |
| `paper` | The row Laplacian has exactly one zero eigenvalue within `ba_qsc_tol`. |
| `no_singleton_sink` (default) | The `paper` condition holds and every sink strongly connected component has at least two nodes. |
| `strong` (also `strongly_connected`) | Every node can reach every other node along directed edges. |

The generator tries up to `ba_max_tries` samples. By default it is best effort:
if none passes, it warns with the selected criterion and returns the final
sample. With zero attempts it returns a zero matrix. These fallbacks need not
satisfy the acceptance rule.

Set `strict_connectivity=True` to reject a failing graph instead. You can also
request an additional `connectivity_requirement`:

- `no_isolated_nodes`: every node has an incident edge.
- `weak`: the graph is connected when edge direction is ignored.
- `strong`: directed paths exist between every ordered pair of nodes.

The additional requirement is checked after generation and does not change the
draw order or trigger resampling. In best-effort mode failure produces a warning;
strict mode raises `ValueError`. Barabasi-Albert still applies its own selected
retry criterion as well. Without an additional requirement, strict mode checks
the random generator's no-isolates rule, the Barabasi-Albert acceptance rule, or
weak connectivity for a supplied matrix. These criteria are distinct: a graph
can pass the no-isolates or sink-component checks without being strongly
connected. `graph_model` accepts `random` and `barabasi_albert`; `erdos_renyi` is
not an implemented alias.

You can also supply a square adjacency matrix. Its size determines the number
of agents:

```python
import numpy as np
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph

adjacency = np.array([
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0],
])

env = NetworkGraph(
    connectivity_matrix=adjacency,
    initial_opinions=np.array([0.1, 0.3, 0.6, 0.8]),
    desired_opinion=1.0,
    max_u=0.1,
    t_campaign=1.0,
    t_s=0.1,
)
opinions, info = env.reset()
```

In the dynamics, `adjacency[i, j]` controls how node `j` contributes to node `i`.
Supplied matrices keep their weights unless `use_weighted_edges=True`, which
replaces entries equal to one with samples from `weight_range`. This option
also samples weights for Barabasi-Albert graphs before row normalization.

## Opinion dynamics

Choose a model with `dynamics_model`:

| Model | Behavior | Main settings |
| --- | --- | --- |
| `laplacian` (also `degroot`) | Continuous consensus using the graph Laplacian. | `t_s` |
| `coca` | Neighbor influence varies with the node's current opinion. | `t_s` |
| `friedkinjohnsen` (also `friedkin_johnsen`, `fj`) | Mixes social influence with a fixed prejudice vector. | `fj_lambda`, `fj_prejudice` |
| `hegselmannkrause` (also `hegselmann_krause`, `hk`) | Averages neighbors whose opinions are close enough. | `hk_epsilon`, `hk_include_self` |
| `nonlinearinfluence` (also `nonlinear_influence`) | Uses a saturating influence function. | `nonlinear_beta`, `t_s` |
| `repulsion` | Attracts similar opinions and pushes distant ones apart. | `repulsion_epsilon`, `repulsion_strength`, `t_s` |

Each environment step applies control once, then runs propagation for
`t_campaign` using intervals of `t_s`. `t_s` must be positive and divide
`t_campaign` evenly. The Friedkin-Johnsen and Hegselmann-Krause models apply a
discrete update at each interval.

Friedkin-Johnsen uses the discrete update
`x_next = fj_lambda * (A @ x) + (1 - fj_lambda) * fj_prejudice` at every internal
interval. Its assumptions are checked explicitly:

- `A` must be a finite, real, nonnegative square matrix. Rows must sum to one,
  checked by summing in float64 with zero relative tolerance. The absolute
  tolerance is `4 * np.finfo(np.float32).eps` (about `4.77e-7`) for float32
  matrices and `1e-12` for other dtypes. This admits small normalization errors
  in single precision, including normalized and softmax-generated rows. Zero
  rows and substantive row-sum errors remain invalid. Validation never changes
  the supplied entries.
- `fj_lambda` must be a scalar or a vector of shape `(num_agents,)`, with finite
  values in `[0,1]`. `fj_prejudice` must be a vector of that shape in `[0,1]`.
- Initial opinions, targets, `max_u`, and control resistance must be finite and
  in `[0,1]`, with per-node vectors of the expected shape. The initial sampling
  range must satisfy `0 <= low <= high <= 1`. Fully resistant nodes are allowed.
- Direct FJ dynamics calls also require a state and control vector of the
  expected shape in `[0,1]`. Propagated states allow `1e-12` roundoff at the
  endpoints; values are never clipped or changed by validation.

These assumptions make both control and social averaging convex combinations,
which preserve opinion bounds in exact arithmetic. Floating-point results can
have endpoint roundoff. Invalid lambda values raise instead of being clipped;
invalid matrices raise instead of being normalized. You may explicitly prepare
a matrix with `normalize_adjacency_matrix` from `graph_utils`, checking that no
zero rows remain.

The declared observation space is exactly `[0,1]`. An endpoint update can
exceed it by roundoff even with a float64 matrix. With float32 rows and
`fj_lambda=1`, positive row-sum errors can accumulate; the next update can
then fail the state validation above. The row-sum tolerance is an acceptance
criterion, not a promise of exact boundedness for an approximately stochastic
matrix. For endpoint-sensitive experiments, prepare and inspect the matrix
in float64 and check the observations. No normalization or clipping is applied
to hide this numerical limitation.

Every generated graph is checked against the FJ requirements too. In particular,
the default random graph is generally unnormalized and can be rejected. Choose
a valid supplied matrix or a Barabasi-Albert configuration with no zero rows;
for example:

```python
env = NetworkGraph(
    graph_model="barabasi_albert", ba_connectivity="strong",
    strict_connectivity=True, dynamics_model="fj", seed=0,
)
```

Generated matrices are validated without automatic repair for FJ. If
`fj_prejudice` is omitted, it is set to the initial opinions of
each episode; a supplied prejudice vector stays fixed across resets.

For bounded-opinion experiments with any model, use finite nonnegative adjacency
weights, initial opinions and targets in `[0,1]`, and controls and resistance in
`[0,1]`. Laplacian consensus preserves this interval in exact arithmetic under
these assumptions. COCA, HK, nonlinear influence, and repulsion retain their
existing output clipping to `[0,1]`. Use nonnegative confidence radii
(`hk_epsilon`, `repulsion_epsilon`), `nonlinear_beta`, and `repulsion_strength`
for their usual interpretations. These are modeling assumptions; outside FJ,
the constructor does not enforce all of them. For the Euler-based models,
choose `t_s` small enough for the accuracy of the experiment.

## Rewards and episode endings

The default reward penalizes distance from the target and the amount of control:

```text
reward = -sum(abs(desired_opinion - opinions)) - control_beta * sum(action)
```

`normalize_reward=True` divides this value by the number of agents.
`use_delta_shaping=True` adds a progress term weighted by `delta_lambda`.
`terminal_reward` adds a bonus for convergence before the step limit.

An episode terminates when the mean opinion is within `opinion_end_tolerance`
of the target. Set `terminate_when_converged=False` to keep running until
`max_steps`, which truncates the episode.

For a per-node target vector, control uses each node's target, while reward
distances and the convergence check use the mean target.

`budget` tracks spending: it does not restrict actions or end an episode.
`info` includes `total_spent`, `remaining_budget` when a budget is set, and
`intermediate_states` for the propagation between control steps. Keep actions
within `action_space`: dynamics clips controls to the allowed range, while
reward costs and spending use the action you supplied.

## Repeating an experiment

The constructor's `seed` controls graph generation and initial opinion sampling.
Two environments built with the same seed and settings start alike.

`reset(seed=0)` restarts opinion sampling on the current graph. Repeating that
reset gives the same opinions; `reset()` continues the random stream. The graph,
Laplacian, and centralities stay fixed. Because construction also draws random
numbers for the graph, a reset seed does not necessarily reproduce the opinions
seen immediately after construction.

If `initial_opinions` is supplied, resets use that vector. To sample opinions
instead, call `reset(randomize_opinions=True)` or
`reset(options={"randomize_opinions": True})`.

## Rendering

```python
fig = env.render(mode="matplotlib", seed=0)
centrality_fig = env.render(mode="centralities")
env.render(mode="human")
```

The first two calls return Matplotlib figures; `"human"` prints the current
opinions and spending. Try `layout_try=1` for another graph layout, or
`randomize_layout=True` to sample a layout. Layout changes leave the simulation
unchanged.

<img src="../../../docs/figures/network_graph/example.png" alt="NetworkGraph visualization" width="400">

## Example trajectories

Opinion trajectories under uniform and near-optimal control,
starting from the same network and initial opinions.

![Opinion trajectories under uniform and near-optimal control](../../../docs/figures/network_graph/uniform_vs_near_optimal.png)
