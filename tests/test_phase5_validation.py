"""Final API checks using supported configurations and independent calculations."""

import ast
import copy
from pathlib import Path
import random
import re

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
from scipy.special import softmax

from rl_envs_forge.envs.acml.car_rental.car_rental import JacksCarRental
from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem
from rl_envs_forge.envs.grid_world.grid_world import Action, GridWorld
from rl_envs_forge.envs.inverted_pendulum.cart_pole.cart_pole import CartPole
from rl_envs_forge.envs.inverted_pendulum.pendulum_disk.pendulum_disk import PendulumDisk
from rl_envs_forge.envs.k_armed_bandit.k_armed_bandit import KArmedBandit
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth
from rl_envs_forge.envs.labyrinth.maze.maze import Maze, MazeFactory
from rl_envs_forge.envs.labyrinth.maze.room import RectangleRoom, RoomFactory
from rl_envs_forge.envs.network_graph.dynamics import friedkin_johnsen_step
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


MODELS = [
    "laplacian", "degroot", "coca", "friedkinjohnsen", "friedkin_johnsen", "fj",
    "hegselmannkrause", "hegselmann_krause", "hk", "nonlinearinfluence",
    "nonlinear_influence", "repulsion",
]
ADJACENCY = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def network(model):
    return NetworkGraph(
        connectivity_matrix=ADJACENCY, dynamics_model=model, seed=0,
        t_campaign=0.2, t_s=0.1, max_steps=6, terminate_when_converged=False,
    )


FACTORIES = [
    pytest.param(lambda: Labyrinth(seed=0), id="labyrinth-full"),
    pytest.param(lambda: Labyrinth(seed=0, state_vision_range=1), id="labyrinth-local"),
    pytest.param(lambda: KArmedBandit(k=3), id="bandit"),
    pytest.param(lambda: KArmedBandit(k=3, arm_params={
        0: {"distribution": "gamma", "shape": 2, "scale": 1},
        1: {"distribution": "beta", "a": 2, "b": 3},
        2: {"distribution": "uniform", "low": -2, "high": 2},
    }), id="bandit-custom"),
    *[pytest.param(lambda p=p: GridWorld(
        rows=3, cols=3, start_state=(1, 1), terminal_states={(2, 2): 7},
        walls={(0, 0)}, p_success=p, episode_length_limit=6,
        slip_distribution={Action.UP: 0.75, Action.DOWN: 0.25},
        special_transitions={((1, 1), Action.RIGHT): ((2, 2), 11)},
    ), id=f"grid-{p}") for p in (0, 0.5, 1)],
    pytest.param(lambda: JacksCarRental(max_cars=3, max_move_cars=1), id="car-rental"),
    pytest.param(lambda: JacksCarRental(max_cars=3, max_move_cars=1,
                                      request_lambda=[0, 0], return_lambda=[0, 0]),
                 id="car-rental-zero-rates"),
    *[pytest.param(lambda p=p: GamblersProblem(goal_amount=10, start_capital=5,
                                             win_probability=p), id=f"gambler-{p}")
      for p in (0, 0.4, 1)],
    pytest.param(CartPole, id="cart-pole"),
    pytest.param(PendulumDisk, id="pendulum-disk"),
    *[pytest.param(lambda model=model: network(model), id=f"network-{model}")
      for model in MODELS],
    pytest.param(lambda: NetworkGraph(num_agents=5, seed=0), id="network-random"),
    pytest.param(lambda: NetworkGraph(num_agents=5, graph_model="barabasi_albert",
                                     ba_connectivity="strong", strict_connectivity=True,
                                     dynamics_model="fj", seed=0), id="network-ba-fj"),
]


@pytest.mark.parametrize("factory", FACTORIES)
def test_representative_gymnasium_checks(factory):
    env = factory()
    try:
        # These classes expose legacy explicit rendering methods, not render_mode.
        check_env(env, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.parametrize("factory", FACTORIES)
def test_seeded_rollouts_spaces_finiteness_and_observation_ownership(factory):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    env = factory()
    try:
        def episode(seed):
            observation, info = env.reset(seed=seed)
            env.action_space.seed(seed)
            assert env.observation_space.contains(observation)
            transitions = [copy.deepcopy((observation, info))]
            if isinstance(observation, np.ndarray):
                state_before = copy.deepcopy(env.state)
                observation.flat[0] = 99
                np.testing.assert_equal(env.state, state_before)
            for _ in range(12):
                action = env.action_space.sample()
                assert env.action_space.contains(action)
                original_action = copy.deepcopy(action)
                transition = env.step(action)
                np.testing.assert_equal(action, original_action)
                observation, reward, terminated, truncated, info = transition
                assert env.observation_space.contains(observation)
                assert np.isfinite(np.asarray(observation)).all()
                assert np.isfinite(reward)
                assert isinstance(terminated, (bool, np.bool_))
                assert isinstance(truncated, (bool, np.bool_))
                transitions.append(copy.deepcopy(transition))
                if isinstance(observation, np.ndarray):
                    # Editing an observation must not change the simulated state.
                    state_before = copy.deepcopy(env.state)
                    observation.flat[0] = 99
                    np.testing.assert_equal(env.state, state_before)
                if terminated or truncated:
                    break
            return transitions

        for seed in (0, 42):
            first = episode(seed)
            env.reset()
            np.testing.assert_equal(episode(seed), first)
    finally:
        env.close()
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state


@pytest.mark.parametrize("sign", [-1, 1])
def test_pendulum_failure_velocity_observation_and_independent_euler_update(sign):
    env = PendulumDisk(episode_length_limit=1)
    initial = np.array([sign * np.pi / 2, sign * 47.0])
    options = {"initial_state": initial}
    observation, _ = env.reset(seed=0, options=options)
    assert env.observation_space.contains(observation)
    observation[1] = 0
    np.testing.assert_array_equal(env.state, initial)
    acceleration = (env.m * env.g * env.l * np.sin(initial[0])
                    - (env.b + env.K**2 / env.R) * initial[1]) / env.J
    expected = np.array([env.normalize_angle(initial[0] + env.tau * initial[1]),
                         initial[1] + env.tau * acceleration])
    observation, _, terminated, truncated, _ = env.step(np.zeros(1, np.float32))
    np.testing.assert_allclose(observation, expected, rtol=0, atol=1e-14)
    assert abs(observation[1]) > 15 * np.pi
    assert terminated and truncated
    assert env.observation_space.contains(observation)
    np.testing.assert_array_equal(options["initial_state"], initial)
    # A failure threshold is not a bound on user-supplied starting velocity.
    assert env.observation_space.contains(env.reset(initial_state=[0, sign * 10000])[0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["normalize", "softmax"])
@pytest.mark.parametrize("seed", range(5))
def test_fj_normalized_matrices_against_independent_reference(dtype, method, seed):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(size=(7, 7)).astype(dtype)
    matrix = raw / raw.sum(axis=1, keepdims=True) if method == "normalize" else softmax(raw, axis=1)
    original = matrix.copy()
    x, lam, prejudice = rng.uniform(0.1, 0.9, size=(3, 7))
    expected = np.array([
        lam[i] * sum(float(matrix[i, j]) * x[j] for j in range(7))
        + (1 - lam[i]) * prejudice[i] for i in range(7)
    ])
    actual = friedkin_johnsen_step(x, adjacency_matrix=matrix, fj_lambda=lam,
                                   prejudice=prejudice)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(matrix, original)
    assert np.all((actual >= 0) & (actual <= 1))
    env = NetworkGraph(connectivity_matrix=matrix, dynamics_model="fj", seed=seed,
                       initial_opinions=x, fj_lambda=lam, fj_prejudice=prejudice,
                       t_s=1, t_campaign=1, terminate_when_converged=False)
    for _ in range(20):
        observation, *_ = env.step(np.zeros(7, dtype=np.float32))
        assert env.observation_space.contains(observation)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("error", [-1e-4, 1e-4, 0.1])
def test_fj_substantive_row_errors_remain_invalid(dtype, error):
    matrix = np.eye(3, dtype=dtype)
    matrix[0, 0] += error
    with pytest.raises(ValueError, match="row-stochastic"):
        friedkin_johnsen_step(np.full(3, 0.4), adjacency_matrix=matrix,
                             fj_lambda=0.9, prejudice=np.full(3, 0.4))


def test_fj_float64_tolerance_is_not_relaxed_to_float32_precision():
    with pytest.raises(ValueError, match="row-stochastic"):
        friedkin_johnsen_step(np.full(2, 0.4), adjacency_matrix=np.eye(2) * (1 + 1e-10),
                             fj_lambda=0.9, prejudice=np.full(2, 0.4))


def test_manual_labyrinth_layout_and_custom_inputs_are_not_mutated():
    layout = np.ones((10, 11), dtype=np.uint8)
    layout[[0, -1], :] = 0
    layout[:, [0, -1]] = 0
    original = layout.copy()
    rewards = {"neutral_reward": 2, "wall_collision_reward": -3, "target_reached_reward": 9}
    env = Labyrinth(maze_layout_predetermined=layout, reward_schema=rewards, seed=0)
    env.reset(seed=0)
    env.set_state((1, 1))
    env.maze.target_position = (1, 3)
    assert env.observation_space.contains(env.state)
    first = env.step(1)
    second = env.step(1)
    assert first[1:4] == (2, False, False)
    assert second[1:4] == (9, True, False)
    np.testing.assert_array_equal(layout, original)
    assert rewards == {"neutral_reward": 2, "wall_collision_reward": -3, "target_reached_reward": 9}


def test_caller_configuration_arrays_and_dictionaries_are_not_mutated():
    inputs = {
        "arms": {0: {"distribution": "normal", "mean": 0, "std": 0,
                     "param_functions": [{"function": lambda t: t, "target_param": "mean"}]}},
        "rewards": {"valid_move": 2, "wall_collision": -4, "out_of_bounds": -5, "default": 0},
        "slip": {Action.UP: 0.25, Action.DOWN: 0.75},
        "special": {((1, 1), Action.RIGHT): ((2, 2), 9)},
        "terminals": {(2, 2): 3}, "walls": {(0, 0)},
        "requests": [2, 3], "returns": [1, 2],
        "adjacency": ADJACENCY.copy(), "initial": np.array([0.2, 0.4, 0.6]),
        "lambda": np.array([0.2, 0.5, 0.8]), "prejudice": np.array([0.4, 0.5, 0.6]),
    }
    snapshot = copy.deepcopy(inputs)
    environments = [
        KArmedBandit(k=1, arm_params=inputs["arms"]),
        GridWorld(rows=3, cols=3, start_state=(1, 1), rewards=inputs["rewards"],
                  slip_distribution=inputs["slip"], special_transitions=inputs["special"],
                  terminal_states=inputs["terminals"], walls=inputs["walls"]),
        JacksCarRental(max_cars=3, max_move_cars=1, request_lambda=inputs["requests"],
                      return_lambda=inputs["returns"]),
        NetworkGraph(connectivity_matrix=inputs["adjacency"], initial_opinions=inputs["initial"],
                     dynamics_model="fj", fj_lambda=inputs["lambda"], fj_prejudice=inputs["prejudice"]),
    ]
    for env in environments:
        env.reset(seed=0)
        action = np.zeros(3, dtype=np.float32) if isinstance(env, NetworkGraph) else 0
        env.step(action)
        env.reset(seed=0)
        env.close()
    np.testing.assert_equal(inputs, snapshot)


@pytest.mark.parametrize("factory", [MazeFactory, Maze, RoomFactory, RectangleRoom])
def test_maze_seed_bounds_are_integers_and_preserve_draw_sequence(factory, monkeypatch):
    original = random.Random(42)
    reference = random.Random(42)

    def strict_randint(low, high):
        # Simulate interpreters that no longer accept float bounds in randrange.
        assert isinstance(low, int) and isinstance(high, int)
        return original.randint(low, high)

    monkeypatch.setattr(random, "randint", strict_randint)
    kwargs = {} if factory is RoomFactory else {"rows": 10, "cols": 10}
    assert factory(**kwargs).seed == reference.randint(0, 1_000_000)


def test_donut_random_draw_uses_integer_bounds_without_changing_draw_order(monkeypatch):
    factory = RoomFactory(seed=42)
    reference = random.Random(42)
    original_randint = factory.py_random.randint

    def strict_randint(low, high):
        assert isinstance(low, int) and isinstance(high, int)
        return original_randint(low, high)

    monkeypatch.setattr(factory.py_random, "randint", strict_randint)
    factory.create_donut_room(rows=8, cols=10)
    reference.randint(1, int(8 // 1.75))
    reference.choice(["rectangle", "oval"])
    reference.choice(["rectangle", "oval"])
    assert factory.py_random.getstate() == reference.getstate()


READMES = [
    "README.md", "rl_envs_forge/envs/labyrinth/README.md",
    "rl_envs_forge/envs/labyrinth/maze/README.md",
    "rl_envs_forge/envs/labyrinth/entities/README.md",
    "rl_envs_forge/envs/labyrinth/display/README.md",
    "rl_envs_forge/envs/k_armed_bandit/README.md", "rl_envs_forge/envs/grid_world/README.md",
    "rl_envs_forge/envs/acml/README.MD", "rl_envs_forge/envs/network_graph/README.md",
    "rl_envs_forge/envs/inverted_pendulum/cart_pole/README.md",
    "rl_envs_forge/envs/inverted_pendulum/pendulum_disk/README.md",
]


class HeadlessExamples(ast.NodeTransformer):
    """Omit only interactive rendering/manual input, leaving simulation intact."""

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "render":
                return ast.parse("(False, None)", mode="eval").body
            if node.func.attr == "human_play":
                return ast.Constant(value=None)
        return self.generic_visit(node)


@pytest.mark.parametrize("readme", READMES)
def test_readme_python_examples_headless(readme):
    path = Path(__file__).resolve().parents[1] / readme
    blocks = re.findall(r"```python\s*\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
    assert blocks
    namespace = {}
    for block in blocks:
        tree = ast.fix_missing_locations(HeadlessExamples().visit(ast.parse(block)))
        exec(compile(tree, str(path), "exec"), namespace)
    for value in namespace.values():
        if isinstance(value, (Labyrinth, KArmedBandit, GridWorld, JacksCarRental,
                              GamblersProblem, CartPole, PendulumDisk, NetworkGraph)):
            value.close()
