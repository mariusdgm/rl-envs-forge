# Tests Usage

Requirements: pytest and pytest-cov

## Running Tests

Install the development environment from the repository root:
```
poetry install
```

Without Poetry, use `python -m pip install ".[test]"`. The compatibility workflow
tests Python 3.10, 3.11, and 3.12 on Linux and Windows with freshly resolved
dependencies. A configured CI job is not evidence that a version passed; see
[release validation](../docs/release_candidate_validation.md) for local results.

Run the tests in the root folder with:

```bash
poetry run pytest tests
```

For a faster test run during development (some repeat tests are marked with the tag 'slow'):
```bash
poetry run pytest tests -m "not slow"
```

To test one environment:
```bash
poetry run pytest tests/envs/network_graph -m "not slow"
```

The default pytest options include module doctests and coverage. The four
`slow` tests generate many mazes; `poetry run pytest tests -m slow` runs just
those checks. The final API audit, mathematical regressions, and README example
smokes are in `tests/test_phase5_validation.py`.

For headless tests in PowerShell:

```powershell
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:MPLBACKEND = 'Agg'
poetry run pytest tests
```

The corresponding shell variables on Linux/macOS are `SDL_VIDEODRIVER=dummy`,
`SDL_AUDIODRIVER=dummy`, and `MPLBACKEND=Agg`. Gymnasium checks omit the legacy
rendering API; rendering tests run separately. README smokes execute Python
blocks in order and omit rendering and manual input.

For release artifacts, run:

```bash
python tests/release_candidate_checks.py --artifacts dist --output docs/_build/release-check --install
```

Use a directory containing exactly one candidate wheel and one candidate sdist,
and choose a new output directory for each run. The checker inspects archive
contents and creates separate virtual environments for the two installs. It
resolves dependencies independently of `poetry.lock` and runs the non-slow tests
against each installed artifact. This command accesses public PyPI; use
`--offline-links /path/to/wheelhouse` to restrict it to local distributions.
