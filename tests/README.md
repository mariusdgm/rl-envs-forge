# Tests Usage

Requirements: pytest and pytest-cov

## Running Tests

Install the development environment from the repository root:
```
poetry install
```

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
