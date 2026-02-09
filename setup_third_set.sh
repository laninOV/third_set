#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Install Python 3.10+ and rerun."
  exit 1
fi

echo "[1/5] Creating virtual environment: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[2/5] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing Python dependencies"
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "[4/5] Installing Playwright Chromium"
python -m playwright install chromium

echo "[5/5] Sanity check"
python -m py_compile "${ROOT_DIR}"/third_set/*.py

echo
echo "Setup completed successfully."
echo
echo "Activate environment:"
echo "  source .venv/bin/activate"
echo
echo "Core commands:"
echo "  python -m third_set.cli live --limit 5 --history 5"
echo "  python -m third_set.cli analyze --match-url \"https://www.sofascore.com/tennis/match/...#id:12345678\" --max-history 5 --history-only --brief"
echo "  python -m third_set.cli probe-upcoming --hours 2 --limit 20"
echo "  python -m third_set.cli tg-bot --history-only"
echo
echo "Telegram (optional):"
echo "  export TELEGRAM_BOT_TOKEN='YOUR_BOT_TOKEN'"
echo "  export TELEGRAM_CHAT_ID='YOUR_CHAT_ID'"
echo "  python -m third_set.cli tg-test"
echo
echo "Locale lock to RU (optional):"
echo "  export THIRDSET_FORCE_RU=1"
