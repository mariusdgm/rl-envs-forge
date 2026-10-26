# Documentation

The package READMEs describe each environment and include runnable examples.
Sphinx builds the API reference from the Python docstrings.

## Build the API reference

From the repository root, install the documentation tools:

```bash
poetry install --extras docs
```

Then run a strict build:

```bash
poetry run python -m sphinx -E -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. On a headless machine, set
`SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`, and `MPLBACKEND=Agg` first.

Without Poetry, install `python -m pip install ".[docs]"` and omit `poetry run`
from the build command. From an extracted source distribution, the same command
installs the local project and its documentation extra.

## Diagrams

The existing figures are included in the source distribution. Regenerating the
UML diagrams is optional and needs Pylint and the Graphviz application; see
[diagram instructions](diagrams/README.md).

## Release validation

[Release candidate validation](release_candidate_validation.md) records local
test results, packaging checks, and the checks still required before release.
