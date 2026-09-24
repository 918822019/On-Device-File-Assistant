#!/usr/bin/env bash
# ============================================================================
# WSL 文件索引层：源目录配置助手
#
# 探测 Windows 侧常见用户目录（Desktop/Downloads/Documents/Pictures/Videos/
# 微信/QQ 等），交互多选后把逗号分隔的多根写入 .env 的 FILE_MEMORY_SOURCE_DIR。
#
# 用法：
#   bash scripts/wsl_sources.sh                 # WSL 下：交互选择 Windows 目录
#   bash scripts/wsl_sources.sh --print         # 只列出探测到的候选，不写 .env
#   bash scripts/wsl_sources.sh /path/a /path/b # 任意平台：直接写入给定目录
#
# 说明：
# - 多根用逗号分隔，路径本身可含空格（如 "WeChat Files"），不要加引号进值里
# - 写入后需重启服务并触发一次重建索引（页面「重建索引」按钮或
#   POST /v1/search-agent/rebuild-index）
# - 扫描排除目录由 FILE_MEMORY_SCAN_EXCLUDE_DIRS 控制（默认已含 node_modules 等）
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

info() { echo "[wsl-sources] $*"; }
fail() { echo "[wsl-sources][ERROR] $*" >&2; exit 1; }

# 与 deploy.sh 相同语义：只替换生效行，无生效行则追加
env_set() {
  local key="$1" value="$2"
  touch .env
  if grep -qE "^[[:space:]]*${key}=" .env; then
    sed -i.bak -E "s|^([[:space:]]*)${key}=.*|\1${key}=${value}|" .env
    rm -f .env.bak
  else
    printf '%s=%s\n' "${key}" "${value}" >> .env
  fi
  info ".env: ${key} 已更新（${#value} 字符）"
}

# ---- 直接传路径模式（任意平台可用）----
if [[ $# -gt 0 && "$1" != "--print" ]]; then
  joined=""
  for p in "$@"; do
    [[ -d "$p" ]] || fail "目录不存在: $p"
    joined="${joined:+${joined},}$p"
  done
  env_set FILE_MEMORY_SOURCE_DIR "${joined}"
  info "重启服务后生效；建议随后触发一次重建索引"
  exit 0
fi

# ---- WSL 探测模式 ----
[[ -d /mnt/c ]] || fail "未找到 /mnt/c：当前不是 WSL 环境。请直接传目录路径（见用法）"

# 找 Windows 用户目录（跳过系统账户目录）
user_dirs=()
for ud in /mnt/c/Users/*; do
  name="$(basename "${ud}")"
  case "${name}" in
    Public|Default*|"All Users"|desktop.ini|Administrator) continue ;;
  esac
  [[ -d "${ud}" ]] && user_dirs+=("${ud}")
done
[[ ${#user_dirs[@]} -gt 0 ]] || fail "/mnt/c/Users 下没有找到用户目录（检查 WSL 自动挂载）"

# 候选目录：常见个人目录 + IM 文件目录（存在才列入）
candidates=()
for ud in "${user_dirs[@]}"; do
  for sub in Desktop Downloads Documents Pictures Videos Music \
             "Documents/WeChat Files" "Documents/xwechat_files" \
             "Documents/Tencent Files" "Documents/DingDing" \
             "Pictures/Camera Roll" "Pictures/Screenshots" \
             "AppData/Roaming/Tencent/WeChat"; do
    dir="${ud}/${sub}"
    [[ -d "${dir}" ]] && candidates+=("${dir}")
  done
done
[[ ${#candidates[@]} -gt 0 ]] || fail "未探测到任何候选目录"

info "探测到 ${#candidates[@]} 个候选目录："
for i in "${!candidates[@]}"; do
  printf "  [%2d] %s\n" "$((i + 1))" "${candidates[$i]}"
done

if [[ "${1:-}" == "--print" ]]; then
  exit 0
fi

echo
read -r -p "[wsl-sources] 选择要纳入索引的编号（空格/逗号分隔，a=全选，q=退出）: " ans
[[ "${ans}" == "q" ]] && { info "已退出，未修改 .env"; exit 0; }

selected=()
if [[ "${ans}" == "a" ]]; then
  selected=("${candidates[@]}")
else
  for token in $(echo "${ans}" | tr ',' ' '); do
    idx=$((token - 1))
    if [[ "${token}" =~ ^[0-9]+$ && ${idx} -ge 0 && ${idx} -lt ${#candidates[@]} ]]; then
      selected+=("${candidates[${idx}]}")
    else
      fail "无效编号: ${token}"
    fi
  done
fi
[[ ${#selected[@]} -gt 0 ]] || fail "未选择任何目录"

joined=""
for p in "${selected[@]}"; do
  joined="${joined:+${joined},}${p}"
done
env_set FILE_MEMORY_SOURCE_DIR "${joined}"

info "已选 ${#selected[@]} 个根目录："
for p in "${selected[@]}"; do info "  - ${p}"; done
echo
info "后续步骤："
info "  1. 重启服务: bash scripts/service.sh restart（或 systemd 同名命令）"
info "  2. 首次全量扫描在启动时自动执行；也可页面点「重建索引」"
info "  3. 跨 9P 扫 /mnt/c 首扫较慢，之后 hash/size 判重前置，增量成本低"
info "  4. 想剪掉更多目录（如 AppData 子树）: 编辑 .env 的 FILE_MEMORY_SCAN_EXCLUDE_DIRS"
