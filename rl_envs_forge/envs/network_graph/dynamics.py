import numpy as np
from scipy.linalg import expm

FJ_ALIASES = frozenset(('friedkinjohnsen', 'friedkin_johnsen', 'fj'))


def validate_unit_interval(value, size, name, allow_scalar=False, atol=0.0):
    """Validate real finite FJ parameters without changing their values."""
    array = np.asarray(value)
    if not (
        (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_)
        and np.isrealobj(array)
        and (array.shape == (size,) or (allow_scalar and array.ndim == 0))
        and np.all(np.isfinite(array))
        and np.all((array >= -atol) & (array <= 1 + atol))
    ):
        scalar_note = ' or a scalar' if allow_scalar else ''
        raise ValueError(f'FJ {name} must have shape ({size},){scalar_note}, with finite values in [0, 1]')


def validate_fj_adjacency(adjacency_matrix, size):
    """Require a nonnegative finite row-stochastic matrix; never normalize it."""
    matrix = np.asarray(adjacency_matrix)
    if not (
        matrix.shape == (size, size) and size > 0
        and (np.issubdtype(matrix.dtype, np.number) or matrix.dtype == np.bool_)
        and np.isrealobj(matrix) and np.all(np.isfinite(matrix))
        and np.all(matrix >= 0)
    ):
        raise ValueError('FJ adjacency must be a square, finite, real, nonnegative matrix')
    # Float64 accumulation exposes rounding already present in float32 entries.
    # Allow a small single-precision normalization error, retaining the existing
    # tolerance for float64 and other storage types. Never change the entries.
    row_atol = 4 * np.finfo(np.float32).eps if matrix.dtype == np.float32 else 1e-12
    if not np.allclose(matrix.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=row_atol):
        raise ValueError(
            f'FJ adjacency must be row-stochastic (row sums 1, atol={row_atol:g}); '
            'no automatic normalization is performed'
        )


def validate_fj_inputs(state, adjacency_matrix, fj_lambda, prejudice):
    """Validate a finite real propagated state and bounded FJ parameters.

    Approximate stochastic matrices need not preserve [0, 1] under repeated
    updates. Initial opinions are bounded separately by NetworkGraph.
    """
    state = np.asarray(state)
    if not (
        state.ndim == 1 and state.size > 0
        and (np.issubdtype(state.dtype, np.number) or state.dtype == np.bool_)
        and np.isrealobj(state) and np.all(np.isfinite(state))
    ):
        raise ValueError('FJ state must be a nonempty one-dimensional finite real vector')
    size = state.size
    validate_fj_adjacency(adjacency_matrix, size)
    validate_unit_interval(fj_lambda, size, 'fj_lambda', allow_scalar=True)
    validate_unit_interval(prejudice, size, 'prejudice')


def apply_impulse_control(current_state, control_action, desired_opinion_vector, control_resistance):
    effective_control = control_action * (1 - control_resistance)
    return effective_control * desired_opinion_vector + (1 - effective_control) * current_state


def laplacian_step(state, *, L, t_s, expL_ts=None, default_t_s=None):
    if expL_ts is not None and default_t_s is not None and abs(t_s - default_t_s) <= 1e-12:
        return expL_ts @ state
    return expm(-L * t_s) @ state


def coca_step(state, *, adjacency_matrix, t_s):
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
    validate_fj_inputs(state, adjacency_matrix, fj_lambda, prejudice)
    lam = np.asarray(fj_lambda, dtype=float)
    if lam.ndim == 0:
        lam = np.full_like(state, float(lam), dtype=float)
    # Preserve the recurrence and its rounding. Report numerical overflow rather
    # than returning an invalid observation or silently repairing the model.
    with np.errstate(over='raise', invalid='raise'):
        social = adjacency_matrix @ state
        result = lam * social + (1.0 - lam) * np.asarray(prejudice, dtype=float)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('FJ update produced a nonfinite state; inspect matrix row sums and rollout length')
    return result


def hegselmann_krause_step(state, *, adjacency_matrix, hk_epsilon, hk_include_self=True):
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


def nonlinear_influence_step(state, *, adjacency_matrix, t_s, nonlinear_beta):
    x = np.asarray(state, dtype=float)
    A = np.asarray(adjacency_matrix, dtype=float)
    beta = float(nonlinear_beta)
    diff = x[None, :] - x[:, None]
    influence = np.tanh(beta * diff)
    delta = (A * influence).sum(axis=1)
    x_next = x + t_s * delta
    return np.clip(x_next, 0.0, 1.0)


def repulsion_step(state, *, adjacency_matrix, t_s, repulsion_epsilon, repulsion_strength):
    x = np.asarray(state, dtype=float)
    A = np.asarray(adjacency_matrix, dtype=float)
    eps = float(repulsion_epsilon)
    rho = float(repulsion_strength)
    diff = x[None, :] - x[:, None]
    transformed = np.where(np.abs(diff) <= eps, diff, -rho * diff)
    delta = (A * transformed).sum(axis=1)
    x_next = x + t_s * delta
    return np.clip(x_next, 0.0, 1.0)


def step_once(model, state, *, adjacency_matrix, L, t_s, desired_opinion_vector=None, control_resistance=None,
              expL_ts=None, default_t_s=None, fj_lambda=None, prejudice=None, hk_epsilon=None,
              hk_include_self=True, nonlinear_beta=None, repulsion_epsilon=None, repulsion_strength=None):
    model = str(model).lower()
    if model in ('laplacian', 'degroot'):
        return laplacian_step(state, L=L, t_s=t_s, expL_ts=expL_ts, default_t_s=default_t_s)
    if model == 'coca':
        return coca_step(state, adjacency_matrix=adjacency_matrix, t_s=t_s)
    if model in FJ_ALIASES:
        if fj_lambda is None:
            raise ValueError('friedkinjohnsen dynamics requires fj_lambda')
        if prejudice is None:
            raise ValueError('friedkinjohnsen dynamics requires prejudice')
        return friedkin_johnsen_step(state, adjacency_matrix=adjacency_matrix, fj_lambda=fj_lambda, prejudice=prejudice)
    if model in ('hegselmannkrause', 'hegselmann_krause', 'hk'):
        if hk_epsilon is None:
            raise ValueError('hegselmannkrause dynamics requires hk_epsilon')
        return hegselmann_krause_step(state, adjacency_matrix=adjacency_matrix, hk_epsilon=hk_epsilon, hk_include_self=hk_include_self)
    if model in ('nonlinearinfluence', 'nonlinear_influence'):
        if nonlinear_beta is None:
            raise ValueError('nonlinearinfluence dynamics requires nonlinear_beta')
        return nonlinear_influence_step(state, adjacency_matrix=adjacency_matrix, t_s=t_s, nonlinear_beta=nonlinear_beta)
    if model == 'repulsion':
        if repulsion_epsilon is None:
            raise ValueError('repulsion dynamics requires repulsion_epsilon')
        if repulsion_strength is None:
            raise ValueError('repulsion dynamics requires repulsion_strength')
        return repulsion_step(state, adjacency_matrix=adjacency_matrix, t_s=t_s,
                              repulsion_epsilon=repulsion_epsilon, repulsion_strength=repulsion_strength)
    raise ValueError(f'Unknown dynamics model: {model}')


def rollout_dynamics(current_state, control_action, *, dynamics_model, adjacency_matrix, L, t_campaign, t_s,
                     desired_opinion_vector, control_resistance, expL_ts=None, default_t_s=None,
                     fj_lambda=None, prejudice=None, hk_epsilon=None, hk_include_self=True,
                     nonlinear_beta=None, repulsion_epsilon=None, repulsion_strength=None):
    if str(dynamics_model).lower() in FJ_ALIASES:
        validate_fj_inputs(current_state, adjacency_matrix, fj_lambda, prejudice)
        size = np.asarray(current_state).size
        validate_unit_interval(control_action, size, 'control_action')
        validate_unit_interval(desired_opinion_vector, size, 'desired_opinion')
        validate_unit_interval(control_resistance, size, 'control_resistance')
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
            nonlinear_beta=nonlinear_beta,
            repulsion_epsilon=repulsion_epsilon,
            repulsion_strength=repulsion_strength,
        )
        intermediate_states.append(current.copy())

    return current, np.array(intermediate_states)
