#!/usr/bin/env bash
# ============================================================================
# [已废弃 → 薄包装] macOS 文件索引层：源目录配置助手
#
# 候选探测（含 IM 沙盒/iCloud 探测与 TCC 可读性检测）/交互多选/写 .env 的
# 逻辑已统一到跨平台的 scripts/sources.py（逻辑层
# src/edge_cloud_agent/source_discovery.py），本脚本仅做转发，
# 保持既有调用方式不破：
#
#   bash scripts/macos_sources.sh                 # 交互选择
#   bash scripts/macos_sources.sh --print         # 只列出探测到的候选，不写 .env
#   bash scripts/macos_sources.sh /path/a /path/b # 直接写入给定目录
#
# 新用法请直接：python scripts/sources.py [--platform macos] [...]
#
# 注意（仍适用）：
# - TCC 隐私权限：macOS 会拦截未授权进程读取 桌面/文稿/下载。候选目录显示
#   ⚠️ 不可读时，到「系统设置 → 隐私与安全性 → 完全磁盘访问权限」授权终端。
# - 不要把根直接设成 $HOME：~/Library 子树极大会拖垮扫描，请选择具体子目录。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-python3}"
command -v "${PY}" >/dev/null 2>&1 || {
  echo "[macos-sources][ERROR] 需要 python3（或设 PYTHON 环境变量指向解释器）" >&2
  exit 1
}

echo "[macos-sources] 已废弃：本脚本自动转发到 python scripts/sources.py --platform macos" >&2
exec "${PY}" "${SCRIPT_DIR}/sources.py" --platform macos "$@"
