#!/usr/bin/env bash
# ============================================================================
# [已废弃 → 薄包装] WSL 文件索引层：源目录配置助手
#
# 候选探测/交互多选/写 .env 的逻辑已统一到跨平台的 scripts/sources.py
# （逻辑层 src/edge_cloud_agent/source_discovery.py），本脚本仅做转发，
# 保持既有调用方式不破：
#
#   bash scripts/wsl_sources.sh                 # WSL 下：交互选择 Windows 目录
#   bash scripts/wsl_sources.sh --print         # 只列出探测到的候选，不写 .env
#   bash scripts/wsl_sources.sh /path/a /path/b # 任意平台：直接写入给定目录
#
# 新用法请直接：python scripts/sources.py [--platform wsl] [...]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-python3}"
command -v "${PY}" >/dev/null 2>&1 || {
  echo "[wsl-sources][ERROR] 需要 python3（或设 PYTHON 环境变量指向解释器）" >&2
  exit 1
}

echo "[wsl-sources] 已废弃：本脚本自动转发到 python scripts/sources.py --platform wsl" >&2
exec "${PY}" "${SCRIPT_DIR}/sources.py" --platform wsl "$@"
