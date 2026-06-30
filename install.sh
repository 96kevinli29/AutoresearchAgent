#!/usr/bin/env bash
# mathagent one-shot installer: sets up an isolated Python env and installs everything.
# Usage:  bash install.sh
set -e
cd "$(dirname "$0")"

echo "==> mathagent installer"

# 1. ensure uv (fast, userspace Python manager — no system Python or root needed)
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv (userspace)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 2. create a Python 3.11 venv and install mathagent (with all extras)
echo "==> creating .venv and installing mathagent…"
export UV_LINK_MODE=copy
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .

# 3. seed a .env if missing
if [ ! -f .env ]; then
  cat > .env <<'ENV'
# Paste your key below (get one at https://openrouter.ai/keys), then save.
OPENROUTER_API_KEY=sk-or-REPLACE_ME
MATHAGENT_MODEL=openrouter/anthropic/claude-sonnet-4.6
ENV
  echo "==> created .env — edit it and add your API key"
fi

cat <<'DONE'

✓ Installed.

Next:
  1) put your API key in  .env
  2) start the web app:    .venv/bin/mathagent serve
     then open            http://127.0.0.1:8000
  or solve from the CLI:   .venv/bin/mathagent solve "Find all real roots of x^2-5x+6=0." --format pdf
DONE
