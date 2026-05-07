# Contributing

Thank you for contributing.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Quality checks

Run before opening a pull request:

```bash
ruff check .
mypy src
pytest
```

## Pull requests

- Keep changes focused and well documented.
- Add or update tests for behavior changes.
- Update `CHANGELOG.md` when appropriate.
