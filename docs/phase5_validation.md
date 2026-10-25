# Phase 5 correctness and release validation

## Executive summary

This phase fixes PendulumDisk observation metadata, NetworkGraph observation
ownership, FJ single-precision row-sum validation, integer random bounds in maze
generation, and an incorrect sprite inclusion path. It adds 103 regression and
audit cases and an opt-in release verification script. Equations, rewards,
stochastic semantics, action encodings, and graph distributions are unchanged.

The final non-slow suite passes **825 tests**, with four slow tests deselected.
The full suite passes **829 tests**, including all four slow tests, with **95%
statement coverage**. Gymnasium checks pass on 28 supported
configurations, and all eight environments pass import/checker/reset/step smokes
from the built wheel. Direct comparison with the starting commit gives exact
matches in **381 valid cases**, including RNG states, and matching rejection of
60 invalid cases.

This is **not yet an unqualified publication-release recommendation**. FJ
endpoint roundoff can violate the exact observation bounds, and newly admitted
float32 matrices can subsequently fail state validation. The strict Sphinx build
also fails. A new release version and a broader Python/platform validation
matrix are needed before publication. These limitations are detailed below;
passing representative checks does not establish correctness for every input.

## Source and execution environment

- Date: 2026-10-25.
- Starting HEAD: **`32303f7a7a2215a1abba06cc788af059f03e3766`**. The starting
  working tree was clean. Results describe that commit plus the uncommitted
  changes listed below; no new commit was created.
- `docs/phase4_validation.md` was absent in this checkout. Its PendulumDisk
  example was reconstructed directly from `reset()` and `step()`.
- Windows runtime: `Windows-10-10.0.26200-SP0`, AMD64; Python **3.10.6**,
  conda-forge build, through the existing Poetry virtual environment.
- Interpreter:
  `C:\Users\Chainsword\AppData\Local\pypoetry\Cache\virtualenvs\rl-envs-forge-78aYwWbq-py3.10\Scripts\python.exe`.
- Poetry 1.6.1; pytest 7.4.4; pytest-cov 4.1.0; Gymnasium 0.29.1; NumPy
  1.26.4; SciPy 1.14.0; NetworkX 3.3; Pygame 2.6.0; Matplotlib 3.9.2;
  Seaborn 0.13.2; setuptools 68.2.2; Sphinx 7.3.7.
- Source package imports resolve to this checkout. Its existing installed
  distribution metadata reports **5.19.2**, while `pyproject.toml` and the newly
  built wheel report **5.23.0**. `rl_envs_forge.__version__` reads installed
  metadata, explaining the development-environment discrepancy. The installed
  wheel reports 5.23.0 correctly. The existing environment was not reinstalled.
- Python **3.12.0** was used for the isolated standard-library reproduction of
  float `randint()` bounds, not for the full library suite.

Sources: [project metadata](../pyproject.toml),
[version lookup](../rl_envs_forge/__init__.py),
[test configuration](../pytest.ini).

## Implemented changes and mathematical checks

### PendulumDisk: failure thresholds are not observation bounds

Minimal public-API reproduction:

```python
import numpy as np
from rl_envs_forge.envs.inverted_pendulum.pendulum_disk.pendulum_disk import PendulumDisk

env = PendulumDisk(episode_length_limit=1)
initial, _ = env.reset(seed=0, options={"initial_state": [np.pi / 2, 47.0]})
observation, reward, terminated, truncated, _ = env.step(np.zeros(1, np.float32))
print(observation, terminated, truncated, env.observation_space.contains(observation))
```

Observed state: approximately `[1.8057963267948969, 47.21744786442546]`;
both ending flags are true. Before the change, observation membership was
false despite the initial state belonging to the declared space. After the
change, membership is true. The returned numerical state and flags are unchanged.

Writing angular velocity as `v`, the implemented Euler update is

```text
v_next = (1 - tau*c)*v + tau*f(angle, voltage)
c = (b + K**2/R)/J
f = (m*g*l*sin(angle) + K*voltage/R)/J
```

A fixed configuration with restricted starting velocities and a stable
timestep admits a velocity estimate. However, the API permits arbitrary
finite starting velocities and configurable timesteps. Even an initially
admissible velocity can overshoot the failure threshold before termination is
reported. There is no shared finite velocity bound for all these configurations.
The observation space now uses angle bounds `[-pi, pi]` and velocity bounds
`[-inf, inf]`, retaining float64. Neither physical state nor dynamics is clipped.

The regression independently calculates both signs of the Euler update,
checks simultaneous failure/timeout, reset copying, unchanged caller options,
and membership for a supplied velocity of magnitude 10,000. Supplied reset
angles are expected to be in `[-pi, pi]`; reset still preserves its input.

Sources: [PendulumDisk constructor/reset/step](../rl_envs_forge/envs/inverted_pendulum/pendulum_disk/pendulum_disk.py),
`test_pendulum_failure_velocity_observation_and_independent_euler_update` in
[the Phase 5 tests](../tests/test_phase5_validation.py).

### FJ: precision-aware validation without modifying coefficients

The FJ equation remains exactly
`lambda * (A @ x) + (1 - lambda) * prejudice`. Validation still requires a
finite, real, square, nonnegative matrix, finite unit-interval parameters,
and correct vector dimensions.

For seven-node matrices generated with `default_rng(0)`, observed maximum row
errors when accumulated in float64 were:

| Preparation | float32 | float64 |
| --- | ---: | ---: |
| Divide uniform samples by their row sums | `5.541369318962097e-8` | `1.1102230246251565e-16` |
| `scipy.special.softmax(..., axis=1)` | `5.960464477539063e-8` | `2.220446049250313e-16` |

The original `1e-12` tolerance rejected both float32 examples. The row-sum
check now uses zero relative tolerance and an absolute tolerance of
`4 * np.finfo(np.float32).eps = 4.76837158203125e-7` for native float32
matrices. Other dtypes retain `1e-12`. This is a small explicit acceptance
budget for single-precision normalization, not a universal error bound for
every normalization algorithm or a way to infer the caller's intent.

Twenty normalized/softmax cases cover two precisions, two preparation methods,
and five seeds. Independent scalar sums agree with the one-step kernel within
`3e-16`; supplied matrices remain exactly unchanged. Twenty-step interior-state
rollouts remain inside the observation space. Row errors of `+/-1e-4` and
`0.1` remain rejected in either precision; float64 error `1e-10` remains rejected.
Existing Phase 4 analytical, alias, invariance, invalid-input, and graph-contract
tests also pass.

**Unresolved numerical contract:** acceptance within a tolerance does not make
the represented matrix exactly stochastic. With the float32 softmax example,
`initial_opinions=np.ones(7)`, `fj_lambda=1`, `t_campaign=t_s=1`, and zero
control, the first observation has maximum `1.0000000596046448`, outside
`Box(0, 1)`. The following step raises the existing FJ state-range error.
The float64 version also produces maximum `1.0000000000000002`, outside the
exact space, but completed 100 tested steps without the range error. These
are measured counterexamples, not fixed boundedness claims. State validation
still allows only `1e-12` endpoint roundoff; it has not been silently relaxed.

Reproduce by constructing the matrix below, passing it to `NetworkGraph` with
the settings above, and taking two zero-control steps:

```python
from scipy.special import softmax
A = softmax(np.random.default_rng(0).uniform(size=(7, 7)).astype(np.float32), axis=1)
```

A future change needs an explicit numerical contract: for example, a documented
error budget for propagated states and observation metadata, or stricter input
preparation requirements. Normalizing coefficients, clipping output, or simply
increasing state tolerance without controlling accumulated error would change
or obscure the scientific model. Those approaches were not implemented.

Sources: `validate_fj_adjacency`, `validate_fj_inputs`,
`friedkin_johnsen_step`, `rollout_dynamics` in
[dynamics.py](../rl_envs_forge/envs/network_graph/dynamics.py);
[FJ regression tests](../tests/test_phase5_validation.py);
[Phase 4 mathematical tests](../tests/envs/network_graph/test_phase4.py).

### NetworkGraph observation ownership

Previously, `observation, _ = env.reset(); observation[0] = -42` changed
`env.opinions[0]` to `-42`. The same alias existed after `step()`. Both methods
now return copies. Tests mutate returned reset/step observations across all
tested model strings and verify that the simulated state is unchanged.
`env.state` and `env.opinions` remain direct state access for existing users.
Numerical values and random draws are unchanged.

Source: [NetworkGraph.reset/step/state](../rl_envs_forge/envs/network_graph/network_graph.py).

### Maze generation and Python integer bounds

`py -3.12 -c "import random; random.randint(0, 1e6)"` raises
`TypeError: 'float' object cannot be interpreted as an integer`. The same
problem affects the integral float produced by `min(rows, cols) // 1.75` in
donut generation. Python 3.10 accepted these values with deprecation warnings.

The four seed-bound occurrences now use integer `1_000_000`; the donut bound
is explicitly converted to an integer. The otherwise discarded donut-width
draw is retained to preserve RNG ordering. Tests exercise the actual constructors
with integer-only random-call checks and compare the complete donut RNG state
with an independently advanced reference generator. No seeding scheme or
generation distribution changed.

Sources: `MazeFactory.__init__`, `Maze.__init__` in
[maze.py](../rl_envs_forge/envs/labyrinth/maze/maze.py);
`RoomFactory.__init__`, `Room.__init__`, `RoomFactory.create_donut_room` in
[room.py](../rl_envs_forge/envs/labyrinth/maze/room.py).

## Per-environment Gymnasium audit

`check_env(..., skip_render_check=True)` ran on 28 configurations. The separate
rollout tests use seeds 0 and 42, repeat each episode after intervening resets,
seed action spaces independently, check finite observations/rewards and space
membership, verify input actions are unchanged, and test observation mutation.
This is representative coverage, not an exhaustive proof over all configurations.

| Environment | Observation and action spaces | Measured checks and episode semantics | Qualification |
| --- | --- | --- | --- |
| Labyrinth | uint8 full grid or local square; `Discrete(4)` actions | Two checker configurations, full/local seeded rollouts; target termination, no internal horizon; manually initialized supplied-layout regression passes | A supplied layout needs start and target assignment after every reset. The full observation aliases a disposable derived cache, but editing it does not change maze/player state. |
| KArmedBandit | `Discrete(k)` observations and actions | Two configurations: default and custom gamma/beta/uniform arms; deterministic seeded rewards; every pull terminates, never truncates | Continuing pulls without reset is a documented legacy workflow; autoreset wrappers would rebuild arms and schedules. |
| GridWorld | Tuple of row/column `Discrete` spaces; `Discrete(4)` actions | Three checker configurations at `p_success=0, 0.5, 1`, custom slips, wall, special reward, terminal destination and horizon; canonical-model regressions pass | Out-of-grid reset starts remain accepted by legacy validation, despite failing observation membership. |
| JacksCarRental | Tuple of two `Discrete(max_cars+1)` spaces; `Discrete(2*max_move_cars+1)` action indices | Two checker configurations, ordinary and zero rates; sampled inventories/rewards finite; both ending flags remain false | A continuing task; experiments need an external horizon. Exact planning and simulator regressions pass. |
| GamblersProblem | Capital `Discrete(goal+1)`; global stake `Discrete(goal+1)` | Three permissive checker configurations at win probabilities 0, 0.4, 1; terminal ruin/goal, never internal truncation | Strict mode needs masked actions. Generic unmasked `check_env` can raise on a state-infeasible stake; helper/model agreement and strict rejection have separate tests. |
| CartPole | Unbounded float64 four-vector; float32 `Box(-3,3,(1,))` actions | Checker and seeded rollouts pass; existing independent failure-only, timeout-only, simultaneous-ending and ordinary-transition tests pass | Infinite-space and action-normalization checker warnings are advisory. Euler accuracy depends on timestep. |
| PendulumDisk | Float64 angle in `[-pi,pi]`, unbounded velocity; float32 `Box(-3,3,(1,))` actions | Checker, seeded rollouts, both velocity-failure signs, reset membership and ending boundaries pass | Supplied reset angle must be within the documented range; no state clipping or reset normalization. |
| NetworkGraph | Float64 `Box(0,1,(n,))`; per-node float32 `Box(0,max_u,(n,))` | Fourteen checker configurations: twelve model strings, random graph and strict BA/FJ; seeded rollouts and independent observations pass | Endpoint roundoff counterexample above prevents a blanket exact-space guarantee. Graph resets preserve the graph. |

All eight also passed a checker/reset/step smoke from the installed wheel.
Legacy `render(mode=...)` interfaces were not presented as modern
`render_mode` constructor support. Rendering is covered separately by existing
tests and the wheel's headless Labyrinth render, which loads both sprites.

Sources: [Phase 5 API tests](../tests/test_phase5_validation.py),
[physical ending tests](../tests/envs/inverted_pendulum/test_termination_semantics.py),
[Gambler feasibility tests](../tests/envs/acml/gambler_problem/test_feasible_actions.py),
[GridWorld stochastic tests](../tests/envs/grid_world/test_stochastic_transitions.py),
[car-rental correctness tests](../tests/envs/acml/car_rental/test_correctness.py).

## Reproducibility and preservation of earlier phases

The opt-in [release checker](../tests/phase5_release_checks.py), function
`compare_reference`, loads source directly from the exact starting Git commit
into separate modules, including the old FJ kernel and maze/room generators.
It shares GridWorld's unchanged action enum between the separately loaded
modules to avoid a Python class-identity artifact.

Across 441 configuration/seed cases:

- **330 valid NetworkGraph cases** match exactly: thirteen model strings
  including uppercase FJ, ten supplied/random/BA graph settings, and three
  seeds. Compared data include graph matrices, constructor opinions, resets,
  up to five steps, rewards, flags, info/intermediate states, and NumPy RNG states.
- **51 valid cases across the other seven environments** match exactly,
  including three maze corridor algorithms, local vision, all three GridWorld
  success probabilities with a special transition, ordinary/zero-rate rentals,
  deterministic gambler extremes, and physical reward modes.
- **60 invalid NetworkGraph cases** retain matching `ValueError` rejections.
  Newly accepted approximate float32 inputs are tested against independent
  arithmetic instead of incorrectly claiming that they had old valid trajectories.

The seeded rollout tests preserve global NumPy/Python RNG states for all 28
representative environment configurations. Existing [seeding tests](../tests/test_seeding.py)
also pass. Standalone maze/room helpers still draw an initial seed from Python's
global RNG when no seed is supplied; that existing behavior is unchanged.
Bandit violin rendering still consumes the environment's reward RNG, as its
README documents. Simulation determinism is scoped to matching settings,
actions, render calls, seeds and numerical stack, not arbitrary platforms.

Caller arrays/dictionaries are unchanged by the tested construction, resets,
steps and parameter schedules. This includes layouts, reward schemas, bandit
schedules, grid walls/rewards/slips/special transitions, car rates, FJ matrices,
initial opinions, lambda and prejudice. This is not a deep-copy guarantee for
all configuration parameters: supplied maze grids and several configuration
objects intentionally retain references. External edits can affect subsequent
use; GridWorld edits require `build_mdp()` as documented.

Phase 1 observation/schema/weight fixes, Phase 2 canonical GridWorld outcomes,
Phase 3 capped-Poisson planning and feasible transfers, and Phase 4 FJ/graph,
feasibility and ending checks all remain covered by the complete suite. No
car-rental, GridWorld or Gambler implementation was edited in this phase.

## Package and documentation validation

`poetry check --lock` passes. Both distributions build locally without network
installation. The obsolete `assets/labyrinth/sprites/*.png` inclusion rule now
names the actual package sprite directory and explicitly covers both formats.
The baseline already included these sprites through package discovery; this
corrects the misleading rule rather than claiming an observed missing-asset bug.

| Artifact | Measured contents |
| --- | --- |
| `dist/rl_envs_forge-5.23.0-py3-none-any.whl` | 57 entries; all environment modules, 11 package READMEs, both Labyrinth PNG sprites, MIT license and distribution metadata |
| `dist/rl_envs_forge-5.23.0.tar.gz` | 56 entries; package sources/READMEs/sprites, root README, license, `pyproject.toml`, `PKG-INFO`; no tests, changelog, lockfile or external documentation figures |

SHA-256 of the measured artifacts:

```text
wheel  156b5f7a18c5dd53f1a0c96d1a97ae6c2bc40166759363ef0f95fcfc220c5526
sdist  426e403322266465f5fe70cf8e1fb4b8c84c3a022956efd2ccca4535b50ee04b
```

`inspect_and_install` creates a fresh temporary venv, installs only the wheel
with `--no-index --no-deps`, and confirms its top-level import/version before
adding dependencies. It then shares the existing Poetry dependency directory
through an explicit `.pth` file. It runs from outside the repository using
`python -I`, asserting that all eight environment module paths belong to the
new venv, not the checkout. All eight checker/reset/step tests and Labyrinth's
headless sprite render pass. All **12 declared runtime dependency versions**
satisfy the wheel's requirements. The final temporary venv was
`C:\Users\Chainsword\AppData\Local\Temp\rl-envs-forge-phase5-ri9rakyq\venv`.

**Isolation limit:** the installed project is isolated; dependencies are reused,
not freshly resolved or installed in a clean dependency environment. This does
not establish a clean online install, transitive dependency consistency for
every supported platform, or Python 3.12+ compatibility. No existing environment
was upgraded or replaced, and no artifact was published.

Additional inspection:

- Metadata permits Python `>=3.10,<4.0`. CI tests only Python 3.10.6. Full
  validation here is also on 3.10.6; newer-version compatibility is not proven.
- Documentation/build packages such as Sphinx, its theme and Graphviz are
  currently required runtime dependencies. No optional extras are defined.
  Matplotlib is imported directly by environments but obtained transitively
  through Seaborn. Dependency organization was not changed.
- Public usage is direct import of each implementation class. Package
  `__init__.py` files do not provide a central environment re-export/registry;
  no `gymnasium.register()` entry points were found. README imports work.
- Python examples from **11 READMEs** execute in order, retaining assertions
  and simulation loops while omitting rendering/manual-input calls. The
  inverted-pendulum overview has no executable Python block. The standalone
  display smoke therefore does not establish interactive/manual-play behavior.
- A normal Sphinx build writes HTML, but strict `-E -W --keep-going` exits **1**
  with **3 errors and 8 warnings**: inherited physical `step()` documentation
  has indentation/list diagnostics, the BA generator docstring has an
  indentation error, and Labyrinth `Action` references are ambiguous. The
  warning log is `docs/_build/phase5/sphinx_warnings.log` (generated/ignored).
- MIT license and author metadata are present. No `CITATION.cff` or
  `codemeta.json` was found. A persistent archival identifier and publication
  citation were not verified. No author/citation information was invented.
- Version remains 5.23.0, and local tag **`v5.23.0` already exists**. Changes
  are recorded under Unreleased. A new release must choose a new version;
  the existing CI tag job rejects a duplicate tag before deployment.

Sources: [metadata](../pyproject.toml), [version lookup](../rl_envs_forge/__init__.py),
[CI workflow](../.github/workflows/citd.yml), [license](../LICENSE),
[Sphinx configuration](conf.py), [release checker](../tests/phase5_release_checks.py),
[README smoke tests](../tests/test_phase5_validation.py).

## Commands and test outcomes

From the repository root, using PowerShell:

```powershell
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:MPLBACKEND = 'Agg'

# Focused regressions and representative check_env/README smokes
poetry run python -B -m pytest tests/test_phase5_validation.py -q --tb=short -o addopts= -p no:cacheprovider

# Non-slow suite, retaining module doctests
poetry run python -B -m pytest tests -m 'not slow' --doctest-modules -q --tb=short -o addopts= -p no:cacheprovider

# All tests, including slow tests, doctests and configured coverage
poetry run python -B -m pytest tests -q --tb=short -p no:cacheprovider

poetry run python -B tests/phase5_release_checks.py --reference 32303f7a7a2215a1abba06cc788af059f03e3766
poetry check --lock

# Process-local Git ownership allowance for Poetry's package file discovery
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = 'D:/Work/repos/RL/rl-envs-forge'
poetry build
poetry run python -B tests/phase5_release_checks.py --wheel dist/rl_envs_forge-5.23.0-py3-none-any.whl

# Expected to fail on the documented pre-existing Sphinx diagnostics
poetry run python -B -m sphinx -E -b html -W --keep-going -w docs/_build/phase5/sphinx_warnings.log docs docs/_build/phase5/html-strict -q

git -c safe.directory=D:/Work/repos/RL/rl-envs-forge diff --check
```

| Check | Passed | Failed | Skipped | Deselected | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Starting HEAD, non-slow | 722 | 0 | 0 | 4 | 13.70 s |
| Final focused Phase 5 file | 103 | 0 | 0 | 0 | 3.00 s |
| Final non-slow suite | 825 | 0 | 0 | 4 | 13.53 s |
| Final full suite with coverage | 829 | 0 | 0 | 0 | 79.79 s; 95% statement coverage |
| Gymnasium configurations, included in focused tests | 28 | 0 | 0 | 0 | Legacy render API excluded explicitly |
| Exact source-reference comparisons | 381 valid + 60 matching rejections | 0 | 0 | 0 | All 441 cases accounted for |
| Wheel environment smokes | 8 | 0 | 0 | 0 | Project isolated; dependencies shared |
| Strict Sphinx build | — | 1 build | — | — | 3 errors, 8 warnings |

`poetry check --lock`, wheel/sdist builds, report-source link checks, new-file
syntax/whitespace checks and `git diff --check` all pass. Coverage is a statement
execution metric, not evidence of a proof of the mathematical assumptions.

The new ownership regression failed on the original implementation, while the
direct pre-fix probes showed the PendulumDisk membership failure and float32
FJ rejection. Final pytest warnings concern deprecated `pkg_resources`, Sphinx
namespace handling, GridWorld's deprecated Matplotlib `tostring_rgb()` call,
headless plotting and checker advice about unbounded/unnormalized spaces.
The non-integer `randrange()` warnings from maze generation are eliminated.
Warnings were not suppressed in the final pytest runs.

## Remaining issues by severity

| Priority | Finding and evidence | Recommended follow-up; implemented status |
| --- | --- | --- |
| High: numerical contract | FJ endpoint observations can exceed exact `[0,1]`; newly admitted float32 rows can trigger the next state-range check | Decide and test a documented finite-precision state/space policy. Do not claim universal boundedness. Documented, deliberately not hidden by clipping or coefficient changes. |
| High: release identity | Source version 5.23.0 already has tag `v5.23.0`; checkout metadata is stale 5.19.2 | Choose a new release version, refresh the development installation and archive the intended source revision. No version/tag changed here. |
| Medium: incomplete validation | `GridWorld().reset(new_start_state=(5,5))` returns an out-of-space state; `GamblersProblem(goal_amount=10,start_capital=11)` likewise; physical reset angles and non-FJ model parameters are not comprehensively checked | Add a separately reviewed validation policy with clear compatibility treatment. These are invalid/precondition-violating inputs accepted by legacy APIs; no speculative semantics change here. |
| Medium: setup/API caveats | Uninitialized supplied Labyrinth layout raises `TypeError: 'NoneType' object is not iterable` on a step; strict Gambler rejects unmasked global stakes; bandit one-step endings differ from the continuing-pull workflow | Preserve and document manual initialization/masks/legacy pull behavior; consider explicit reset configuration and wrapper guidance in a separate API change. Manual workflow and masks remain intact. |
| Medium: documentation build | Strict Sphinx fails with measured errors/warnings | Correct inherited/owned docstring formatting and qualify `Action` references; rerun strict Sphinx. No broad documentation refactor here. |
| Medium: compatibility evidence | Python `<4` claim exceeds measured 3.10 validation; clean dependency resolution and platform matrix were not run | Test supported Python/platform combinations and a fresh dependency install before release. The Python 3.12 integer-bound failure is fixed, but full 3.12 compatibility is not claimed. |
| Low: packaging/dependency maintenance | Source distribution omits tests/changelog; package README images point outside the wheel; direct Matplotlib import relies on a transitive requirement; deprecated rendering/resource APIs remain | Consider source-test inclusion, documentation distribution, direct/optional dependency organization, and migration of deprecated APIs in follow-up work. No dependencies changed. |
| Low: scientific release metadata | No machine-readable software citation or verified archival identifier | Add author-reviewed citation/archive metadata and reproducible experiment artifacts. This phase runs no RL training or comparative benchmark and makes no novelty/performance claim. |

The invalid-reset and uninitialized-layout examples above were executed. The
strict Gambler limitation was reproduced with start capital 1, goal 10 and
action-space seed 0: unmasked `check_env` attempted stake 9 and raised
`ValueError`. Other forward-looking recommendations are proposals, not
implemented features or claims of failed behavior on every supported setup.

## Changed files and compatibility

| Files | Purpose |
| --- | --- |
| [PendulumDisk implementation](../rl_envs_forge/envs/inverted_pendulum/pendulum_disk/pendulum_disk.py) | Correct observation velocity bounds only |
| [NetworkGraph](../rl_envs_forge/envs/network_graph/network_graph.py), [dynamics](../rl_envs_forge/envs/network_graph/dynamics.py) | Copy reset/step observations; precision-aware FJ row acceptance |
| [maze.py](../rl_envs_forge/envs/labyrinth/maze/maze.py), [room.py](../rl_envs_forge/envs/labyrinth/maze/room.py) | Integer bounds, preserving all existing draws |
| [pyproject.toml](../pyproject.toml) | Correct explicit sprite path; no version/dependency changes |
| [Labyrinth README](../rl_envs_forge/envs/labyrinth/README.md), [Bandit README](../rl_envs_forge/envs/k_armed_bandit/README.md), [GridWorld README](../rl_envs_forge/envs/grid_world/README.md), [ACML README](../rl_envs_forge/envs/acml/README.MD) | Manual layout workflow and parameter preconditions |
| [NetworkGraph README](../rl_envs_forge/envs/network_graph/README.md), [PendulumDisk README](../rl_envs_forge/envs/inverted_pendulum/pendulum_disk/README.md), [CartPole README](../rl_envs_forge/envs/inverted_pendulum/cart_pole/README.md) | Observation ownership/bounds, numerical caveats and model assumptions |
| [Phase 5 tests](../tests/test_phase5_validation.py), [release checker](../tests/phase5_release_checks.py), [testing instructions](../tests/README.md) | Repeatable API, mathematical, documentation, source-regression and wheel checks |
| [CHANGELOG](../CHANGELOG.md), [this report](phase5_validation.md) | Unreleased changes and measured validation evidence |

Intentional compatibility differences:

1. Code that edits a NetworkGraph reset/step observation to control the live
   state must instead use the existing state interface. Observation values,
   dtype and trajectories are preserved.
2. PendulumDisk's declared velocity space is now unbounded. Wrappers that use
   finite observation limits for scaling must select explicit scaling; actual
   physical observations, rewards and ending flags are unchanged.
3. Approximate native float32 FJ rows previously rejected can now construct and
   run, subject to the endpoint caveat. Other dtype tolerances remain unchanged.
4. Integer-bound corrections make previously failing calls usable on Python
   versions rejecting float bounds, without changing the accepted integer range
   or draw ordering on Python 3.10.

README structure, tables of contents, images and the ignored experimental
notebook workflow are preserved. Generated artifacts remain local/ignored;
no commit, push, reset, rebase, dependency upgrade or publication was performed.

## Publication-oriented release recommendation

The tested stack supports continued controlled experiments and the targeted
fixes are numerically conservative. Before presenting the library as generally
Gymnasium-compliant scientific software, resolve the FJ finite-precision
state/observation contract, select a new release identity, obtain a clean strict
documentation build, and validate the intended Python/platform/dependency
matrix. A publication should state these tested assumptions and archive the
source/tests and experiment configurations; the current tests establish
implementation evidence, not scientific novelty or comparative performance.
