"""Observation and reward contracts from the independent issue verification."""

import numpy as np
import pytest

from rl_envs_forge.envs.grid_world.grid_world import Action, GridWorld
from rl_envs_forge.envs.k_armed_bandit.k_armed_bandit import KArmedBandit
from rl_envs_forge.envs.labyrinth.constants import PATH, PLAYER, TARGET, WALL
from rl_envs_forge.envs.labyrinth.labyrinth import Labyrinth
from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


def manual_labyrinth(shape=(5, 6), **kwargs):
    layout = np.full(shape, WALL, dtype=np.int64)
    layout[1:-1, 1:-1] = PATH
    env = Labyrinth(maze_layout_predetermined=layout, seed=0, **kwargs)
    # Supplied layouts retain the existing explicit initialization workflow.
    assert env.player.position is None
    assert env.maze.target_position is None
    env.set_state((1, 1))
    env.maze.target_position = (1, 3)
    return env, layout


@pytest.mark.parametrize("vision_range", [None, 1, 2])
def test_labyrinth_generated_observations_match_space(vision_range):
    env = Labyrinth(seed=0, state_vision_range=vision_range)
    for action in range(env.action_space.n):
        observation, _ = env.reset(seed=0)
        assert observation.dtype == np.uint8
        assert env.observation_space.contains(observation)
        grid = env.maze.grid.copy()
        observation, *_ = env.step(action)
        assert observation.dtype == np.uint8
        assert env.observation_space.contains(observation)
        np.testing.assert_array_equal(env.maze.grid, grid)
        if vision_range is None:
            expected = grid.copy()
            expected[env.maze.target_position] = TARGET
            expected[env.player.position] = PLAYER
            np.testing.assert_array_equal(observation, expected)


@pytest.mark.parametrize("shape", [(5, 6), (12, 10)])
@pytest.mark.parametrize("vision_range", [None, 1])
def test_labyrinth_supplied_layout_uses_effective_dimensions(shape, vision_range):
    env, layout = manual_labyrinth(shape, state_vision_range=vision_range)
    assert (env.rows, env.cols) == shape
    expected_shape = shape if vision_range is None else (3, 3)
    assert env.observation_space.shape == expected_shape
    assert env.observation_space.contains(env.state)
    for done in (False, True):
        observation, reward, terminated, truncated, _ = env.step(Action.RIGHT)
        assert env.observation_space.contains(observation)
        assert reward == (10 if done else -0.01)
        assert terminated == done
        assert not truncated
    np.testing.assert_array_equal(env.maze.grid, layout)
    assert layout[1, 1] == layout[1, 3] == PATH


def test_labyrinth_custom_schema_is_copied_and_used_for_each_outcome():
    schema = {
        "neutral_reward": -0.1,
        "wall_collision_reward": -7,
        "target_reached_reward": 5,
    }
    env, _ = manual_labyrinth(reward_schema=schema)
    schema["wall_collision_reward"] = -99
    for action, expected_reward, expected_done in (
        (Action.UP, -7, False),
        (Action.RIGHT, -0.1, False),
        (Action.RIGHT, 5, True),
    ):
        observation, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(observation)
        assert reward == expected_reward
        assert terminated == expected_done
        assert not truncated


@pytest.mark.parametrize("schema", [None, {}])
def test_labyrinth_default_and_empty_schemas_keep_default_rewards(schema):
    env, _ = manual_labyrinth(reward_schema=schema)
    assert env.step(Action.UP)[1] == -1
    assert env.step(Action.RIGHT)[1] == -0.01
    assert env.step(Action.RIGHT)[1] == 10


@pytest.mark.parametrize("k", [1, 2, 5])
def test_bandit_space_contains_unchanged_arm_index_observations(k):
    env = KArmedBandit(k=k, seed=0)
    assert env.observation_space.n == k
    for arm in range(k):
        observation, info = env.reset(seed=0)
        assert (observation, info) == (0, {})
        assert env.observation_space.contains(observation)
        observation, _, terminated, truncated, _ = env.step(arm)
        assert observation == arm
        assert env.observation_space.contains(observation)
        assert terminated and not truncated


@pytest.mark.parametrize(
    "model",
    ["laplacian", "coca", "friedkinjohnsen", "hegselmannkrause",
     "nonlinearinfluence", "repulsion"],
)
def test_network_observation_space_describes_float64_state(model):
    # Row-stochastic adjacency also meets the FJ model's mathematical premise.
    adjacency = (np.ones((3, 3)) - np.eye(3)) / 2
    env = NetworkGraph(
        connectivity_matrix=adjacency,
        initial_opinions=[0.1, 0.5, 0.9],
        dynamics_model=model,
        terminate_when_converged=False,
        seed=0,
    )
    assert env.observation_space.dtype == np.float64
    observation, _ = env.reset(seed=0)
    assert observation.dtype == env.opinions.dtype == np.float64
    assert env.observation_space.contains(observation)
    for value in (0.025, 0.075, 0.05):
        observation, *_ = env.step(np.full(3, value, dtype=np.float32))
        assert observation.dtype == env.opinions.dtype == np.float64
        assert env.observation_space.contains(observation)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.bool_, np.float32, np.float64])
@pytest.mark.parametrize("seed", [0, 42])
def test_network_weighted_adjacency_preserves_weights_and_rng_order(dtype, seed):
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=dtype)
    original = adjacency.copy()
    rng = np.random.default_rng(seed)
    expected_dtype = dtype if np.issubdtype(dtype, np.floating) else np.float64
    expected = adjacency.astype(expected_dtype)
    expected[adjacency == 1] = rng.uniform(0.1, 1.0, size=3)
    expected_opinions = rng.uniform(0.0, 1.0, size=3)

    env = NetworkGraph(
        connectivity_matrix=adjacency, use_weighted_edges=True, seed=seed
    )
    assert env.connectivity_matrix.dtype == expected_dtype
    np.testing.assert_array_equal(env.connectivity_matrix, expected)
    np.testing.assert_array_equal(env.opinions, expected_opinions)
    np.testing.assert_equal(env.np_random.bit_generator.state, rng.bit_generator.state)
    np.testing.assert_array_equal(adjacency, original)


@pytest.mark.parametrize("weighted", [False, True])
def test_network_weighting_leaves_nonunit_float_edges_unchanged(weighted):
    adjacency = np.array([[0, 0.25, 1], [1, 0, 0], [0.75, 1, 0]], dtype=np.float32)
    env = NetworkGraph(
        connectivity_matrix=adjacency, use_weighted_edges=weighted, seed=0
    )
    mask = adjacency != 1 if weighted else np.ones_like(adjacency, dtype=bool)
    np.testing.assert_array_equal(env.connectivity_matrix[mask], adjacency[mask])
    assert env.connectivity_matrix.dtype == np.float32


def test_network_unweighted_integer_adjacency_remains_integer():
    adjacency = np.array([[0, 1], [1, 0]], dtype=np.int64)
    env = NetworkGraph(connectivity_matrix=adjacency, seed=0)
    np.testing.assert_array_equal(env.connectivity_matrix, adjacency)
    assert env.connectivity_matrix.dtype == adjacency.dtype


def test_network_laplacian_trajectory_retains_float64_precision():
    env = NetworkGraph(
        connectivity_matrix=np.array([[0, 0.7], [0.7, 0]]),
        initial_opinions=[0.123456789123, 0.876543210876],
        control_resistance=[0.0, 0.25],
        t_campaign=0.2, t_s=0.1, terminate_when_converged=False, seed=0,
    )
    expected = np.array(env.initial_opinions, dtype=np.float64)
    env.reset(seed=0)
    for values in ([0.02, 0.07], [0.09, 0.01], [0.04, 0.06]):
        action = np.array(values, dtype=np.float32)
        control = action.astype(np.float64) * np.array([1.0, 0.75])
        controlled = control + (1 - control) * expected
        # Closed-form two-agent consensus: the mean stays fixed and the
        # difference decays as exp(-2 * edge_weight * campaign_duration).
        mean = controlled.mean()
        difference = (controlled[0] - controlled[1]) / 2 * np.exp(-2 * 0.7 * 0.2)
        expected = mean + np.array([difference, -difference])
        observation, *_ = env.step(action)
        np.testing.assert_allclose(observation, expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize(
    "action,wall,boundary_start",
    [(Action.UP, (0, 1), (0, 1)),
     (Action.RIGHT, (1, 2), (1, 2)),
     (Action.DOWN, (2, 1), (2, 1)),
     (Action.LEFT, (1, 0), (1, 0))],
)
@pytest.mark.parametrize("custom_rewards", [False, True])
def test_grid_distinguishes_wall_and_boundary_rewards(action, wall, boundary_start, custom_rewards):
    rewards = {"wall_collision": -7, "out_of_bounds": -3} if custom_rewards else None
    for start, walls, expected_reward in (
        ((1, 1), {wall}, -7 if custom_rewards else -1),
        (boundary_start, set(), -3 if custom_rewards else -1),
    ):
        env = GridWorld(
            rows=3, cols=3, start_state=start, terminal_states={},
            walls=walls, rewards=rewards, seed=0,
        )
        assert env.calculate_reward(start, action, start) == expected_reward
        assert env.calculate_transition(start, action) == (start, expected_reward, False)
        assert env.mdp[(start, action)] == [(start, expected_reward, False, 1.0)]
        observation, reward, terminated, truncated, _ = env.step(action)
        assert observation == start
        assert reward == expected_reward
        assert not terminated and not truncated


def test_grid_regular_terminal_and_special_rewards_keep_precedence():
    env = GridWorld(
        rows=3, cols=3, start_state=(1, 1), terminal_states={(1, 2): 10},
        walls={(0, 1)}, rewards={"valid_move": -0.2, "wall_collision": -7, "out_of_bounds": -3},
        special_transitions={((1, 1), Action.UP): ((1, 1), 17)}, seed=0,
    )
    assert env.calculate_transition((1, 1), Action.DOWN) == ((2, 1), -0.2, False)
    assert env.calculate_transition((1, 1), Action.RIGHT) == ((1, 2), 10, True)
    assert env.step(Action.UP)[0:4] == ((1, 1), 17, False, False)


def test_grid_missing_collision_reward_keeps_default_fallback():
    env = GridWorld(
        walls={(0, 1)}, rewards={"valid_move": -0.2}, seed=0,
    )
    assert env.step(Action.RIGHT)[1] == -1
    assert env.step(Action.UP)[1] == -1


def test_grid_collision_rewards_preserve_slip_outcomes_and_probabilities():
    env = GridWorld(
        rows=3, cols=3, start_state=(1, 1), terminal_states={}, walls={(1, 2)},
        rewards={"wall_collision": -7, "out_of_bounds": -3, "valid_move": -0.1},
        p_success=0.5, seed=0,
    )
    assert env.calculate_outcomes((1, 1), Action.RIGHT) == [
        ((1, 1), -7, False, 0.5),
        ((0, 1), -0.1, False, 1 / 6),
        ((2, 1), -0.1, False, 1 / 6),
        ((1, 0), -0.1, False, 1 / 6),
    ]
    expected = np.zeros(9)
    expected[[1, 3, 7]] = 1 / 6
    expected[4] = 0.5
    np.testing.assert_array_equal(env.P[4, :, Action.RIGHT], expected)
