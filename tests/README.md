# Tests Usage

Requirements: pytest and pytest-cov

Run the tests in the root folder with:

```bash
pytest tests
```

For a faster test run during development (some repeat tests are marked with the tag 'slow'):
```bash
pytest tests -m "not slow"
```