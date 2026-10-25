"""Opt-in release checks; does not install into the development environment.

Run from the repository root with Poetry. See docs/phase5_validation.md for
the reference commit and commands. No network, commits, or publishing occur.
"""

import argparse
import contextlib
import copy
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import venv
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "rl_envs_forge.envs."
PATHS = {
    "Labyrinth": "labyrinth.labyrinth", "KArmedBandit": "k_armed_bandit.k_armed_bandit",
    "GridWorld": "grid_world.grid_world", "JacksCarRental": "acml.car_rental.car_rental",
    "GamblersProblem": "acml.gambler_problem.gambler_problem",
    "CartPole": "inverted_pendulum.cart_pole.cart_pole",
    "PendulumDisk": "inverted_pendulum.pendulum_disk.pendulum_disk",
    "NetworkGraph": "network_graph.network_graph",
}


def reference_module(commit, suffix):
    full_name = PREFIX + suffix
    path = full_name.replace(".", "/") + ".py"
    source = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{commit}:{path}"],
        cwd=ROOT, text=True,
    )
    module = types.ModuleType(full_name)
    module.__package__ = full_name.rpartition(".")[0]
    exec(compile(source, f"{commit}:{path}", "exec"), module.__dict__)
    return module


def compare_reference(commit):
    modules = {name: reference_module(commit, suffix) for name, suffix in PATHS.items()}
    dynamics = reference_module(commit, "network_graph.dynamics")
    for name in ("FJ_ALIASES", "rollout_dynamics", "validate_fj_adjacency",
                 "validate_fj_inputs", "validate_unit_interval"):
        setattr(modules["NetworkGraph"], name, getattr(dynamics, name))
    room = reference_module(commit, "labyrinth.maze.room")
    maze = reference_module(commit, "labyrinth.maze.maze")
    maze.RoomFactory, maze.Room = room.RoomFactory, room.Room
    modules["Labyrinth"].MazeFactory, modules["Labyrinth"].Maze = maze.MazeFactory, maze.Maze
    current = {name: getattr(importlib.import_module(PREFIX + suffix), name)
               for name, suffix in PATHS.items()}
    from rl_envs_forge.envs.grid_world.grid_world import Action
    # A separately executed module defines a distinct Enum class. Share its
    # unchanged public action encoding when passing actions between versions.
    modules["GridWorld"].Action = Action

    graph_cases = [
        {"connectivity_matrix": np.roll(np.eye(4), 1, axis=1)},
        {"connectivity_matrix": np.full((4, 4), 0.25)},
        {"num_agents": 2, "bidirectional_prob": 1},
        *[{"num_agents": 6, "bidirectional_prob": p} for p in (0, 0.5, 1)],
        {"num_agents": 6, "graph_connection_distribution": "normal"},
        {"num_agents": 6, "graph_connection_distribution": "exponential"},
        *[{"num_agents": 6, "graph_model": "barabasi_albert", "use_weighted_edges": w,
           "ba_connectivity": "strong", "strict_connectivity": True} for w in (False, True)],
    ]
    models = ["laplacian", "degroot", "coca", "friedkinjohnsen", "friedkin_johnsen", "fj", "FJ",
              "hegselmannkrause", "hegselmann_krause", "hk", "nonlinearinfluence",
              "nonlinear_influence", "repulsion"]
    cases = [("NetworkGraph", dict(graph, dynamics_model=model, t_campaign=0.2, t_s=0.1,
                                   max_steps=5, terminate_when_converged=False))
             for graph in graph_cases for model in models]
    cases += [("Labyrinth", {"maze_corridor_algorithm": algorithm})
              for algorithm in ("prim", "astar", "gbfs")]
    cases += [("Labyrinth", {"state_vision_range": 1}), ("KArmedBandit", {"k": 3})]
    cases += [("GridWorld", {"p_success": p, "special_transitions": {
        ((0, 0), Action.RIGHT): ((3, 4), 7)}}) for p in (0, 0.5, 1)]
    cases += [("JacksCarRental", {}), ("JacksCarRental", {
        "max_cars": 3, "request_lambda": [0, 0], "return_lambda": [0, 0]})]
    cases += [("GamblersProblem", {"start_capital": 50, "win_probability": p}) for p in (0, 0.4, 1)]
    cases += [(name, {"continuous_reward": continuous, "nonlinear_reward": True})
              for name in ("CartPole", "PendulumDisk") for continuous in (False, True)]
    accepted = rejected = 0
    for name, kwargs in cases:
        for seed in (0, 1, 42):
            parameters = dict(kwargs)
            if name in ("Labyrinth", "KArmedBandit", "GridWorld", "NetworkGraph"):
                parameters["seed"] = seed
            environments, failures = [], []
            for cls in (getattr(modules[name], name), current[name]):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        environments.append(cls(**copy.deepcopy(parameters)))
                    failures.append(None)
                except ValueError as exc:
                    failures.append((type(exc).__name__, str(exc)))
            assert failures[0] == failures[1], (name, parameters, failures)
            if failures[0] is not None:
                rejected += 1
                continue
            old, new = environments
            if name == "NetworkGraph":
                np.testing.assert_array_equal(old.connectivity_matrix, new.connectivity_matrix)
                np.testing.assert_array_equal(old.opinions, new.opinions)
            np.testing.assert_equal(old.reset(seed=seed), new.reset(seed=seed))
            rng = np.random.default_rng(seed + 99)
            for _ in range(5):
                if name == "NetworkGraph":
                    action = rng.uniform(0, 0.1, new.num_agents).astype(np.float32)
                elif name in ("CartPole", "PendulumDisk"):
                    action = rng.uniform(-3, 3, 1).astype(np.float32)
                elif name == "JacksCarRental":
                    action = new.max_move_cars
                elif name == "GamblersProblem":
                    action = 1
                else:
                    action = 1
                before = copy.deepcopy(old.step(action))
                after = copy.deepcopy(new.step(action))
                np.testing.assert_equal(before, after)
                np.testing.assert_equal(old.np_random.bit_generator.state, new.np_random.bit_generator.state)
                if before[2] or before[3]:
                    break
            old.close()
            new.close()
            accepted += 1
    return {"reference_commit": commit, "exact_matching_valid_cases": accepted,
            "matching_rejected_cases": rejected, "total_cases": len(cases) * 3}


WHEEL_SMOKE = r'''
import importlib, importlib.metadata as md, json, pathlib, sys
import numpy as np
from gymnasium.utils.env_checker import check_env
classes = json.loads(sys.argv[1])
root = pathlib.Path(sys.prefix).resolve()
results = {}
for name, suffix in classes.items():
    module = importlib.import_module('rl_envs_forge.envs.' + suffix)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root), module.__file__
    kwargs = {'seed': 0} if name in ('Labyrinth', 'NetworkGraph') else {}
    env = getattr(module, name)(**kwargs)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    action = np.zeros(env.action_space.shape, np.float32) if name in (
        'CartPole', 'PendulumDisk', 'NetworkGraph') else 0
    obs, reward, *_ = env.step(action)
    assert env.observation_space.contains(obs) and np.isfinite(reward)
    if name == 'Labyrinth':
        env.render(sleep_time=0)
    env.close()
    results[name] = 'import, checker, reset, step passed'
import rl_envs_forge
package = pathlib.Path(rl_envs_forge.__file__).parent
assert pathlib.Path(rl_envs_forge.__file__).resolve().is_relative_to(root)
for sprite in ('flag.png', 'player.png'):
    assert (package / 'envs/labyrinth/display/sprites' / sprite).is_file()
from packaging.requirements import Requirement
dependencies = {}
for item in md.requires('rl-envs-forge'):
    req = Requirement(item)
    if req.marker is None or req.marker.evaluate():
        installed = md.version(req.name)
        dependencies[req.name] = {'version': installed, 'satisfies': installed in req.specifier}
assert all(item['satisfies'] for item in dependencies.values()), dependencies
print(json.dumps({'wheel_version': rl_envs_forge.__version__, 'module_origin': str(package),
                  'checks': results, 'declared_dependencies': dependencies}))
'''


def inspect_and_install(wheel):
    wheel = wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for sprite in ("flag.png", "player.png"):
            assert f"rl_envs_forge/envs/labyrinth/display/sprites/{sprite}" in names
        assert any(n.endswith(".dist-info/LICENSE") for n in names)
        wheel_files, wheel_readmes = len(names), sum(n.lower().endswith("readme.md") for n in names)
    sdist = wheel.with_name(wheel.name.split("-py3-")[0] + ".tar.gz")
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        sdist_files = len(names)
        assert any(n.endswith("/README.md") for n in names)
        assert any(n.endswith("/pyproject.toml") for n in names)
    temporary = Path(tempfile.mkdtemp(prefix="rl-envs-forge-phase5-"))
    isolated = temporary / "venv"
    # Avoid ensurepip's bundled setuptools shadowing the existing dependencies.
    venv.EnvBuilder(with_pip=False).create(isolated)
    python = isolated / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install",
                    "--disable-pip-version-check", "--no-input", "--no-index", "--no-deps", str(wheel)],
                   cwd=temporary, check=True)
    subprocess.run([str(python), "-I", "-c", "import rl_envs_forge; print(rl_envs_forge.__version__)"],
                   cwd=temporary, check=True)
    # Reuse existing dependencies explicitly; no package or dependency upgrades.
    site = isolated / ("Lib/site-packages" if sys.platform == "win32" else
                       f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    import gymnasium
    (site / "phase5_shared_dependencies.pth").write_text(
        str(Path(gymnasium.__file__).resolve().parents[1]) + "\n", encoding="utf-8")
    completed = subprocess.run([str(python), "-I", "-c", WHEEL_SMOKE, json.dumps(PATHS)],
                               cwd=temporary, text=True, capture_output=True)
    print(completed.stderr, file=sys.stderr)
    if completed.returncode:
        print(completed.stdout)
        completed.check_returncode()
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    results.update({"wheel_files": wheel_files, "wheel_package_readmes": wheel_readmes,
                    "sdist_files": sdist_files, "temporary_venv": str(isolated),
                    "dependency_isolation": "Existing Poetry dependencies shared via explicit .pth; no fresh resolution"})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    result = {}
    if args.reference:
        result["numerical_regression"] = compare_reference(args.reference)
    if args.wheel:
        result["package"] = inspect_and_install(args.wheel)
    print(json.dumps(result, indent=2))
