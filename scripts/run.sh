#!/usr/bin/env bash
# 启动 edge-cloud agent 服务。
# 显式使用项目自带 .venv 的解释器，避免依赖 PATH 而跑到系统 Python。
set -euo pipefail

# 无论从哪里调用，都定位到仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"

# 选择解释器：优先项目 .venv，其次已激活的虚拟环境
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
else
  echo "[run.sh] 找不到虚拟环境。请先执行: make install" >&2
  echo "[run.sh]   （或 uv venv .venv --python 3.11 && uv pip install -r requirements.txt）" >&2
  exit 1
fi

# 拒绝非 venv 解释器，防止再次生成 cpython-3xx 混乱缓存
if ! "${PYTHON}" -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)'; then
  echo "[run.sh] ${PYTHON} 不是虚拟环境解释器，已中止。" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[run.sh] python   = $("${PYTHON}" -V 2>&1)"
echo "[run.sh] venv     = ${PYTHON}"
echo "[run.sh] PYTHONPATH = ${PYTHONPATH}"
echo "[run.sh] listening = http://${HOST}:${PORT}"

exec "${PYTHON}" -m uvicorn edge_cloud_agent.main:app --host "${HOST}" --port "${PORT}"
