"""Opt-in bitwise snapshot of explicitly isolated package sources.

Run with ``python -I tests/pre_release_numerical_checks.py ROOT OUTPUT``.
ROOT must contain the entire rl_envs_forge package, not selected baseline modules.
See docs/pre_release_corrections_validation.md for the capture/comparison procedure.
Observation-space metadata is deliberately excluded: H2 changes that metadata.
"""


def main():
    import argparse
    import hashlib
    import importlib.metadata
    import json
    from pathlib import Path
    import sys
    import warnings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))

    import numpy as np
    from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph
    from rl_envs_forge.envs.acml.car_rental.car_rental import JacksCarRental
    from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem


    def encode(value):
        if isinstance(value, np.ndarray):
            return {"dtype": value.dtype.str, "shape": list(value.shape),
                    "bytes": value.tobytes().hex()}
        if isinstance(value, (float, np.floating)):
            return {"float_hex": float(value).hex()}
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if value is None or isinstance(value, str):
            return value
        raise TypeError(type(value))


    cases = {}


    def record(name, value):
        assert name not in cases
        cases[name] = encode(value)


    def network_case(name, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            env = NetworkGraph(seed=42, max_steps=20, terminate_when_converged=False,
                               **kwargs)
        initial = (env.connectivity_matrix.copy(), env.L.copy(), env.expL_ts.copy(),
                   env.centralities.copy(), env.opinions.copy(),
                   env.np_random.bit_generator.state)
        reset = env.reset(seed=2718)
        reset_rng = env.np_random.bit_generator.state
        # Exercise the uncached matrix-exponential path as well as normal steps.
        override = env.compute_dynamics(env.opinions.copy(),
                                        np.zeros(env.num_agents, np.float32),
                                        t_campaign=.4, t_s=.2)
        env.action_space.seed(314)
        transitions = []
        for step in range(12):
            action = (np.zeros(env.num_agents, np.float32) if step % 3 == 0
                      else env.action_space.sample())
            transitions.append((action, env.step(action),
                                env.np_random.bit_generator.state,
                                env.action_space.np_random.bit_generator.state))
        record(name, (kwargs, initial, reset, reset_rng, override, transitions,
                      [str(item.message) for item in caught]))


    for model in ("laplacian", "degroot"):
        for n in (2, 5, 6):
            adjacency = np.ones((n, n), np.float64) - np.eye(n)
            network_case(f"{model}/float64/complete/{n}", dynamics_model=model,
                         connectivity_matrix=adjacency)
            network_case(f"{model}/float64/endpoint/{n}", dynamics_model=model,
                         connectivity_matrix=adjacency, initial_opinions=np.ones(n),
                         control_resistance=np.zeros(n, np.float32), max_u=1)
        network_case(f"{model}/float64/directed", dynamics_model=model,
                     connectivity_matrix=np.array([[0., .7, .1], [.2, 0., .4], [.8, .3, 0.]]),
                     control_resistance=np.array([0., .25, .75]),
                     initial_opinions=[.123456789123, .5, .876543210876],
                     t_s=.1, t_campaign=.2)
        network_case(f"{model}/float64/weighted", dynamics_model=model,
                     connectivity_matrix=np.array([[0., 1., .2], [1., 0., 1.], [.8, 1., 0.]]),
                     use_weighted_edges=True)

    for model in ("fj", "friedkinjohnsen", "friedkin_johnsen"):
        for lam in (1., np.array([0., .5, 1.])):
            network_case(f"{model}/float64/{lam}", dynamics_model=model,
                         connectivity_matrix=np.array([[.5, .25, .25], [.25, .5, .25], [.25, .25, .5]]),
                         fj_lambda=lam, fj_prejudice=np.array([.2, .5, .8]),
                         control_resistance=np.zeros(3, np.float32))

    for distribution in ("uniform", "normal", "exponential"):
        for bidirectional in (0., .5, 1.):
            network_case(f"random/{distribution}/{bidirectional}", num_agents=8,
                         graph_connection_distribution=distribution,
                         bidirectional_prob=bidirectional)
    for criterion in ("paper", "no_singleton_sink", "strong"):
        for weighted in (False, True):
            network_case(f"ba/{criterion}/{weighted}", num_agents=8,
                         graph_model="barabasi_albert", ba_connectivity=criterion,
                         use_weighted_edges=weighted)
    network_case("ba/fallback", num_agents=8, graph_model="barabasi_albert", ba_max_tries=0)
    network_case("fj/ba/strong", num_agents=8, graph_model="barabasi_albert",
                 ba_connectivity="strong", dynamics_model="fj")

    for model in ("coca", "hk", "nonlinear_influence", "repulsion"):
        network_case(f"unchanged/{model}", dynamics_model=model,
                     connectivity_matrix=np.array([[0., .5, .5], [.5, 0., .5], [.5, .5, 0.]]))

    for rates in (([0, 0], [0, 0]), ([2, 4], [3, 1])):
        env = JacksCarRental(max_cars=3, max_move_cars=1,
                            request_lambda=rates[0], return_lambda=rates[1],
                            init_state_option="equal")
        reset = env.reset(seed=42)
        rng_before = env.np_random.bit_generator.state
        model = env.build_transition_model()
        rows = [(key, model[key], model.expected_reward(*key)) for key in model]
        legacy = env.build_mdp()
        rng_after = env.np_random.bit_generator.state
        transitions = []
        for action in (0, 1, 2, 2, 1, 0, 2, 0, 1, 2):
            transitions.append((env.step(action), env.np_random.bit_generator.state))
        record(f"car/{rates}", (reset, rng_before, rows, legacy, rng_after, transitions))

    for goal in (5, 10):
        for probability in (0., .4, 1.):
            env = GamblersProblem(goal_amount=goal, start_capital=2,
                                  win_probability=probability)
            reset = env.reset(seed=42)
            rng_before = env.np_random.bit_generator.state
            helpers = [(state, env.feasible_actions(state), env.action_mask(state))
                       for state in range(goal + 1)]
            model = env.build_mdp()
            rng_after = env.np_random.bit_generator.state
            transitions = []
            for _ in range(10):
                result = env.step(1)
                transitions.append((result, env.np_random.bit_generator.state))
                if result[2]:
                    break
            record(f"gambler/{goal}/{probability}",
                   (reset, rng_before, helpers, model, rng_after, transitions))

    sources = {}
    for name, module in sorted(sys.modules.items()):
        if name == "rl_envs_forge" or name.startswith("rl_envs_forge."):
            path = Path(module.__file__).resolve()
            relative = path.relative_to(root)  # Fails if any helper leaks from another checkout.
            sources[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    result = {"cases": cases, "provenance": {
        "root": str(root), "package_sources": sources, "python": sys.version,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "networkx", "gymnasium")},
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "isolated_package_modules": len(sources),
                      "output": str(args.output),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
