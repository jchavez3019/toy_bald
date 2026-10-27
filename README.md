# Toy BALD

Toy experimentation with Bayesian Active Learning by Disagreements (BALD). 

> Status: Work in progress.

## What is here

The repository is currently organized around Jupyter notebooks.

* `configs/` - Hydra configuration trees.
* `notebooks/` - exploratory notebooks
* `docs/adr/` - Architecture Decision Records (start here to understand design choices)
* `tests/` - the pytest suite.

## Prerequisites

- Conda (Miniconda or Anaconda) for environment management.
- Python 3.13 (provisioned by the conda environment).
- Git.
- For GPU training: an NVIDIA GPU with CUDA 12.1 drivers. CPU-only use is fine for dataset synthesis and most tests.

## Setup

The steps below create the environment, install the package in editable mode, and configure credentials.

```bash
# 1. Clone
git clone git@github.com:jchavez3019/toy_bald.git
cd toy_bald

# 2. Create and activate the conda environment (Python 3.13 + core deps)
conda env create -f environment.yaml
conda activate toy-bald

# 3. Install the project in editable mode so `import e2e_rag` works
pip install -e .

# 4. Configure credentials: copy the template and fill in the values you use
cp env.template .env
# then edit .env (e.g. NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL)
```

The `.env` file is git-ignored. Only populate the providers you actually use; see [env.template](env.template) for the full list of recognized variables.

### Optional: agent debugging with mcp-debugpy

[jchavez3019/mcp-debugpy](https://github.com/jchavez3019/mcp-debugpy) is my public fork of the [`mcp-debugpy`](https://github.com/markomanninen/mcp-debugpy) project. It is an MCP server that lets an AI agent in Cursor launch a script under `debugpy`, set breakpoints, step through code, and inspect variables. It is not installed by default. To enable it, run one of the following inside the activated `toy_bald` environment, after step 3:

```bash
# Option A: fresh copy from GitHub, through this project's optional `debug` extra
pip install -e '.[debug]'

# Option B: reuse an existing local clone of the fork (editable, so edits to the clone apply immediately)
pip install -e /path/to/mcp-debugpy
```

Either option installs the `mcp-debug-server` command into the `toy_bald` environment. The repository already contains [.cursor/mcp.json](.cursor/mcp.json), which tells Cursor to start that server with `conda run -n toy_bald` from the project root. Reload Cursor after installing (or toggle `agentDebug` under *Cursor Settings → MCP*), and the agent gains tools such as `dap_launch`, `dap_locals`, `dap_step_over`, and `dap_continue`. If you named the conda environment something other than `toy_bald`, edit `.cursor/mcp.json` to match.


Hydra writes run outputs relative to `cwd`, so `"cwd": "."` (the project root) keeps them under `outputs/` as with a normal run.

## Common commands

These are the developer quality gates and entry points. The quality gates are mandated by [ADR 0001](docs/adr/0001-code-maintainability.md).

| Task | Command |
| --- | --- |
| Format | `ruff format .` |
| Lint | `ruff check .` (add `--fix` to autofix) |
| Type-check | `pyright` |
| Test | `pytest` |
| Test with coverage floor | `pytest --cov=e2e_rag --cov-report=term-missing --cov-fail-under=70` |

Note: `pytest-cov` must be installed for the coverage command (it is a planned addition to the environment; install with `pip install pytest-cov` in the meantime).

## Documentation

Design decisions live as Architecture Decision Records under [docs/adr/](docs/adr/README.md):

- [ADR 0001 - Code Maintainability and Quality Gates](docs/adr/0001-code-maintainability.md)

## Roadmap

The near-term direction, roughly in order:

- Migrate the temporal-synthesis components from the notebook into `src/e2e_rag/utils/synthesize_datasets/temporal/`.
- Add a Hydra-driven dataset-synthesis entry script that writes JSONL + manifest artifacts.
- Build the training-side `torch` dataset that reads the JSONL and tokenizes per model.
- Add `pytest-cov` to the environment and wire the quality gates into CI.
