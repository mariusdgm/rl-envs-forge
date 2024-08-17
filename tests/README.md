# Tests Usage

Requirements: pytest and pytest-cov

## Running Tests

Open the poetry shell with:
```
poetry shell
```

Run the tests in the root folder with:

```bash
pytest tests
```

For a faster test run during development (some repeat tests are marked with the tag 'slow'):
```bash
pytest tests -m "not slow"
```

To test a single test file:
```bash
pytest tests/envs/network_graph -m "not slow"
```