import numpy as np
from scipy.linalg import expm


def apply_impulse_control(current_state, control_action, desired_opinion_vector, control_resistance):
    effective_control = control_action * (1 - control_resistance)
    return effective_control * desired_opinion_vector + (1 - effective_control) * current_state


def laplacian_step(state, *, L, t_s, expL_ts=None, default_t_s=None):
    """Advance opinions by one substep under continuous-time Laplacian consensus.

    This is the linear consensus / DeGroot-family propagation used by the
    environment's ``"laplacian"`` and ``"degroot"`` dynamics names.
    The update applies the matrix exponential of the graph Laplacian:

        x_next = exp(-L * t_s) @ x

    When a cached ``expL_ts`` is provided and ``t_s`` matches ``default_t_s``,
    the cached propagator is reused to avoid recomputing the matrix exponential.

    References:
        - DeGroot, M. H. (1974). "Reaching a Consensus."
          Journal of the American Statistical Association.
        - In this codebase, the implementation uses a continuous-time Laplacian
          propagator rather than DeGroot's original discrete-time row-stochastic
          averaging step.

    Args:
        state (np.ndarray): Current opinion vector of shape ``(N,)``.
        L (np.ndarray): Graph Laplacian matrix of shape ``(N, N)``.
        t_s (float): Propagation substep duration.
        expL_ts (np.ndarray | None): Optional cached value of
            ``exp(-L * default_t_s)``. If provided and ``t_s`` matches
            ``default_t_s``, this cached matrix is used.
        default_t_s (float | None): Time step associated with ``expL_ts``.

    Returns:
        np.ndarray: Updated opinion vector of shape ``(N,)``.

    Notes:
        - This function does not clip the output to ``[0, 1]``.
        - The caller is responsible for ensuring ``L`` is consistent with the
          intended adjacency / influence matrix.
    """
    if expL_ts is not None and default_t_s is not None and abs(t_s - default_t_s) <= 1e-12:
        return expL_ts @ state
    return expm(-L * t_s) @ state


def coca_step(state, *, adjacency_matrix, t_s):
    """Advance opinions by one substep under COCA / CODA-style dynamics.

    This update implements the nonlinear continuous-opinion dynamics used in
    this project under the name ``"coca"``. It is inspired by the
    Continuous Opinions and Discrete Actions (CODA) framework, where agents
    maintain continuous internal opinions while social influence is mediated by
    action-level neighbor effects.

    In this implementation, each node updates according to:

        x_next = x + t_s * (x * (1 - x) / deg) * (A @ x - deg * x)

    where:
        - ``A`` is the adjacency / influence matrix,
        - ``deg[i]`` is the row-sum (out-degree) of node ``i``,
        - ``x_i (1 - x_i)`` is a nonlinear gain term that reduces responsiveness
          near the extremes 0 and 1.

    Reference:
        - Chowdhury, M., Morărescu, I. C., Martin, S., and Srikant, R. (2016).
          "Continuous opinions and discrete actions in social networks:
          a multi-agent system approach."

    Args:
        state (np.ndarray): Current opinion vector of shape ``(N,)``.
        adjacency_matrix (np.ndarray): Weighted or binary adjacency matrix of
            shape ``(N, N)``. Row ``i`` defines the neighbors that influence
            node ``i``.
        t_s (float): Propagation substep duration.

    Returns:
        np.ndarray: Updated opinion vector of shape ``(N,)``, clipped to the
        interval ``[0, 1]``.

    Notes:
        - Nodes with zero degree receive no propagation update.
        - Opinions are clipped to ``[0, 1]`` after the update.
        - Compared with linear consensus / DeGroot-style dynamics, this model
          is nonlinear and can exhibit richer behavior such as clustering,
          dissensus, and slowed movement near the extremes.
    """
    A = adjacency_matrix
    deg = A.sum(axis=1)
    inv_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]

    x = state
    neighbor_sum = A @ x
    sum_diff = neighbor_sum - deg * x
    gain = x * (1.0 - x) * inv_deg
    x_next = x + t_s * gain * sum_diff
    return np.clip(x_next, 0.0, 1.0)


def friedkin_johnsen_step(state, *, adjacency_matrix, fj_lambda, prejudice):
    """Advance opinions by one substep under Friedkin-Johnsen dynamics.

    Friedkin-Johnsen extends linear social influence by combining:
        - a social averaging term from neighbors, and
        - an anchoring term pulling each node toward a fixed prejudice
          (often its initial opinion)

    The update is:

        x_next = lambda * (A @ x) + (1 - lambda) * prejudice

    where ``lambda`` may be either a scalar or a per-node vector in ``[0, 1]``.

    References:
        - Friedkin, N. E., and Johnsen, E. C. (1990).
          "Social Influence and Opinions."
          Journal of Mathematical Sociology.

    Args:
        state (np.ndarray): Current opinion vector of shape ``(N,)``.
        adjacency_matrix (np.ndarray): Row-stochastic or otherwise normalized
            social influence matrix of shape ``(N, N)``.
        fj_lambda (float | np.ndarray): Influenceability parameter. May be:
            - a scalar applied to all nodes, or
            - a vector of shape ``(N,)`` with per-node influenceability.
            Values are clipped to ``[0, 1]``.
        prejudice (np.ndarray): Fixed prejudice / anchoring vector of shape
            ``(N,)``. In many uses, this is the initial opinion vector.

    Returns:
        np.ndarray: Updated opinion vector of shape ``(N,)``.

    Notes:
        - ``fj_lambda = 1`` reduces to pure social averaging.
        - ``fj_lambda = 0`` yields full anchoring to ``prejudice``.
        - This function does not clip the output; callers should ensure the
          inputs are in the intended opinion range if bounded opinions are
          required.
    """
    lam = np.asarray(fj_lambda, dtype=float)
    if lam.ndim == 0:
        lam = np.full_like(state, float(lam), dtype=float)
    lam = np.clip(lam, 0.0, 1.0)
    social = adjacency_matrix @ state
    return lam * social + (1.0 - lam) * np.asarray(prejudice, dtype=float)


def hegselmann_krause_step(state, *, adjacency_matrix, hk_epsilon, hk_include_self=True):
    """Advance opinions by one substep under Hegselmann-Krause dynamics.

    Hegselmann-Krause is a bounded-confidence model: node ``i`` only averages
    over neighbors whose opinions are within a confidence radius
    ``hk_epsilon`` of its own current opinion.

    In this implementation:
        - candidate neighbors come from the graph structure in
          ``adjacency_matrix``
        - only neighbors satisfying ``abs(x_j - x_i) <= hk_epsilon`` are used
        - if ``hk_include_self`` is True, node ``i`` is included in its own
          confidence set

    References:
        - Hegselmann, R., and Krause, U. (2002).
          "Opinion Dynamics and Bounded Confidence Models, Analysis, and
          Simulation."
          Journal of Artificial Societies and Social Simulation.

    Args:
        state (np.ndarray): Current opinion vector of shape ``(N,)``.
        adjacency_matrix (np.ndarray): Binary or weighted adjacency matrix of
            shape ``(N, N)``. Row ``i`` defines the candidate neighbors that
            may influence node ``i``.
        hk_epsilon (float): Confidence threshold. Only neighbors within this
            absolute opinion distance are considered.
        hk_include_self (bool): Whether each node should include itself in the
            averaging set. Defaults to True.

    Returns:
        np.ndarray: Updated opinion vector of shape ``(N,)``, clipped to the
        interval ``[0, 1]``.

    Notes:
        - If no confident neighbors are available, the node keeps its current
          opinion.
        - If the graph is weighted, relative neighbor weights are preserved
          within the confident subset.
        - This implementation is graph-constrained HK: unlike the classical
          fully-connected formulation, candidate interaction partners are first
          filtered by the provided adjacency matrix.
    """
    x = np.asarray(state, dtype=float)
    A = np.asarray(adjacency_matrix, dtype=float)
    n = x.shape[0]
    x_next = np.empty_like(x)
    eps = float(hk_epsilon)

    for i in range(n):
        nbrs = np.flatnonzero(A[i] > 0)
        if hk_include_self:
            nbrs = np.unique(np.concatenate(([i], nbrs)))
        if nbrs.size == 0:
            x_next[i] = x[i]
            continue

        close = nbrs[np.abs(x[nbrs] - x[i]) <= eps]
        if hk_include_self and i not in close:
            close = np.unique(np.concatenate((close, [i])))
        if close.size == 0:
            x_next[i] = x[i]
            continue

        # If weighted graph, preserve relative weights on confident neighbors.
        if i in close:
            peer_nodes = close[close != i]
            if peer_nodes.size == 0:
                x_next[i] = x[i]
                continue
            peer_weights = A[i, peer_nodes]
            if np.sum(peer_weights) <= 0:
                x_next[i] = np.mean(x[close])
            else:
                self_weight = 1.0 if hk_include_self else 0.0
                weights = np.concatenate(([self_weight], peer_weights))
                values = np.concatenate(([x[i]], x[peer_nodes]))
                x_next[i] = np.average(values, weights=weights)
        else:
            weights = A[i, close]
            if np.sum(weights) <= 0:
                x_next[i] = np.mean(x[close])
            else:
                x_next[i] = np.average(x[close], weights=weights)

    return np.clip(x_next, 0.0, 1.0)


def step_once(model, state, *, adjacency_matrix, L, t_s, desired_opinion_vector=None, control_resistance=None,
              expL_ts=None, default_t_s=None, fj_lambda=None, prejudice=None, hk_epsilon=None,
              hk_include_self=True):
    model = str(model).lower()
    if model in ('laplacian', 'degroot'):
        return laplacian_step(state, L=L, t_s=t_s, expL_ts=expL_ts, default_t_s=default_t_s)
    if model == 'coca':
        return coca_step(state, adjacency_matrix=adjacency_matrix, t_s=t_s)
    if model in ('friedkinjohnsen', 'friedkin_johnsen', 'fj'):
        if fj_lambda is None:
            raise ValueError('friedkinjohnsen dynamics requires fj_lambda')
        if prejudice is None:
            raise ValueError('friedkinjohnsen dynamics requires prejudice')
        return friedkin_johnsen_step(state, adjacency_matrix=adjacency_matrix, fj_lambda=fj_lambda, prejudice=prejudice)
    if model in ('hegselmannkrause', 'hegselmann_krause', 'hk'):
        if hk_epsilon is None:
            raise ValueError('hegselmannkrause dynamics requires hk_epsilon')
        return hegselmann_krause_step(state, adjacency_matrix=adjacency_matrix, hk_epsilon=hk_epsilon, hk_include_self=hk_include_self)
    raise ValueError(f'Unknown dynamics model: {model}')


def rollout_dynamics(current_state, control_action, *, dynamics_model, adjacency_matrix, L, t_campaign, t_s,
                     desired_opinion_vector, control_resistance, expL_ts=None, default_t_s=None,
                     fj_lambda=None, prejudice=None, hk_epsilon=None, hk_include_self=True):
    controlled_state = apply_impulse_control(current_state, control_action, desired_opinion_vector, control_resistance)
    intermediate_states = [controlled_state.copy()]
    num_steps = int(round(t_campaign / t_s))
    current = controlled_state.copy()

    for _ in range(num_steps):
        current = step_once(
            dynamics_model,
            current,
            adjacency_matrix=adjacency_matrix,
            L=L,
            t_s=t_s,
            desired_opinion_vector=desired_opinion_vector,
            control_resistance=control_resistance,
            expL_ts=expL_ts,
            default_t_s=default_t_s,
            fj_lambda=fj_lambda,
            prejudice=prejudice,
            hk_epsilon=hk_epsilon,
            hk_include_self=hk_include_self,
        )
        intermediate_states.append(current.copy())

    return current, np.array(intermediate_states)
