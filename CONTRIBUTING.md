# Contributing to PromptStrike

Thanks for your interest. This document covers dev setup, running tests, and the PR process.

---

## Dev setup

```bash
git clone https://github.com/Pyhroff/promptstrike
cd promptstrike

# create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install in editable mode with dev extras
pip install -e ".[dev]"

# configure environment
cp .env.example .env
# add GROQ_API_KEY to .env
```

---

## Running tests

```bash
# all tests
pytest tests/ -v

# single file
pytest tests/test_tap.py -v

# with coverage (install pytest-cov first)
pytest tests/ --cov=promptstrike --cov-report=term-missing
```

Tests are fully mocked — no API keys required to run them.

---

## Code style

PromptStrike uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# check
ruff check promptstrike/

# fix in-place
ruff check --fix promptstrike/

# format
ruff format promptstrike/
```

The line length is **100**. Target is Python 3.11+.

---

## Project layout

```
promptstrike/
├── adapters/   — provider clients (Groq, OpenAI, Ollama)
├── api/        — FastAPI dashboard backend + static HTML
├── core/       — attack algorithms (PAIR, TAP) + judge
├── report/     — HTML report generator + Jinja2 templates
└── storage/    — SQLite persistence layer

tests/
├── test_pair.py
├── test_tap.py
├── test_api.py
└── test_ci.py
```

---

## Adding a new adapter

1. Create `promptstrike/adapters/<provider>.py`.
2. Subclass `BaseAdapter` from `adapters/base.py` — implement `chat()` and `name`.
3. Add it to `adapters/__init__.py`.
4. Handle the new provider string in `cli.py::_make_target_adapter()` and `api/server.py::_run_scan()`.

---

## Adding a new attack algorithm

1. Create `promptstrike/core/<algorithm>.py`.
2. Define a `run_<algorithm>()` async function with an `on_event` callback for live streaming.
3. Define a result dataclass with `status: AttackStatus`, `calls_used: int`, `winning_prompt: str`, `final_score: int`.
4. Export from `core/__init__.py`.
5. Wire it into `cli.py` (`scan` command `--algo` flag) and `api/server.py`.
6. Add unit tests in `tests/test_<algorithm>.py` using the same mock adapter pattern as `test_tap.py`.

---

## Pull request guidelines

- **One concern per PR.** Bug fix, new adapter, new algorithm — not mixed.
- **Tests required.** New core logic needs unit tests. Target: all mocked, no API keys.
- **No secrets.** Never commit `.env` files or API keys. The `.gitignore` covers common cases.
- **Describe the change.** PR description should state what changed and why — the code shows the how.
- **Authorized use only.** Contributions must be consistent with the project's security research scope described in [SECURITY.md](SECURITY.md).
