# Contributing

Thanks for your interest in `mimo-sdk-py`! This is a community-maintained,
unofficial SDK and contributions are welcome.

## Quick start

```bash
git clone https://github.com/amilaa-0/mimo-sdk-py
cd mimo-sdk-py

# Recommended: uv (fast)
uv venv .venv --python python3.10
uv pip install --python .venv/bin/python -e ".[cli,dev]"

# Or with stdlib venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[cli,dev]"
```

## Running checks

```bash
.venv/bin/pytest -q                       # unit tests
.venv/bin/ruff check src tests examples   # lint
.venv/bin/mypy src                        # type-check
```

All three should be green before opening a PR.

## Project layout

```
src/mimo/
  client.py           — synchronous high-level client
  async_client.py     — asynchronous mirror
  _transport.py       — HTTP, retry, SSE, error mapping (shared)
  _request_builder.py — pure functions that turn kwargs into JSON bodies
  types.py            — Pydantic v2 schemas
  catalog.py          — built-in model metadata
  vision.py           — image encoding helpers
  errors.py           — exception hierarchy
  cli.py              — Typer-based CLI

tests/                — unit tests using respx for transport mocking
examples/             — runnable demos for every public API
```

## Conventions

- Targets **Python 3.9+**. Don't use 3.10 syntax (`X | Y`, `match`).
- Use `from __future__ import annotations` at the top of every module.
- Run `ruff check --fix` before committing.
- New public API needs a corresponding test and an example.
- Update `CHANGELOG.md` under `[Unreleased]`.

## Reporting issues

Please include:

1. SDK version (`python -c "import mimo; print(mimo.__version__)"`)
2. Python version
3. Minimal reproducer
4. Full traceback (with `XIAOMI_API_KEY` redacted!)

## License

By contributing you agree your changes are licensed under the project's
[MIT License](LICENSE).
