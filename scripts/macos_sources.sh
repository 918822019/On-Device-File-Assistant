#!/usr/bin/env bash
# ============================================================================
# macOS 文件索引层：源目录配置助手
#
# 探测 macOS 常见个人目录（桌面/下载/文稿/图片/影片/音乐/iCloud Drive）与
# IM 沙盒目录（微信/QQ/企业微信/钉钉），交互多选后把逗号分隔的多根写入
# .env 的 FILE_MEMORY_SOURCE_DIR。
#
# 用法：
#   bash scripts/macos_sources.sh                 # 交互选择
#   bash scripts/macos_sources.sh --print         # 只列出探测到的候选，不写 .env
#   bash scripts/macos_sources.sh /path/a /path/b # 直接写入给定目录
#
# 注意：
# - TCC 隐私权限：macOS 会拦截未授权进程读取 桌面/文稿/下载。若候选目录显示
#   ⚠️ 不可读，请到「系统设置 → 隐私与安全性 → 完全磁盘访问权限」把你运行
#   本脚本/后端的终端（Terminal/iTerm/PyCharm 等）加进去并重启终端。
# - iCloud Drive 中未下载到本地的文件是 .icloud 占位符，不在后缀白名单内，
#   会被自然跳过（不会触发下载）。
# - 不要把根直接设成 $HOME：~/Library 子树极大会拖垮扫描，请选择具体子目录。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

info() { echo "[macos-sources] $*"; }
fail() { echo "[macos-sources][ERROR] $*" >&2; exit 1; }

# 与 deploy.sh / wsl_sources.sh 相同语义：只替换生效行，无生效行则追加
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

# ---- 直接传路径模式 ----
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

# ---- 探测模式 ----
[[ "$(uname -s)" == "Darwin" ]] || fail "当前不是 macOS。WSL 请用 scripts/wsl_sources.sh，其他平台直接传目录路径"

candidates=()

add_candidate() {
  # 显式 return 0：set -e 下函数末命令 `[[ -d ]] && …` 在目录不存在时返回 1，
  # 会把整个脚本静默杀掉（顶层 AND 列表可豁免，函数返回值不豁免）
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    candidates+=("${dir}")
  fi
  return 0
}

# 标准个人目录
for sub in Desktop Downloads Documents Pictures Movies Music \
           "Pictures/Screenshots" "Library/Mobile Documents/com~apple~CloudDocs"; do
  add_candidate "${HOME}/${sub}"
done

# 微信（3.x/4.x 沙盒布局）：优先更深的 MessageTemp（避开缓存子树），
# 深层 glob 不中时退回 Application Support 根，由排除目录剪枝兜底
wechat_as="${HOME}/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat"
wechat_deep_found=0
if [[ -d "${wechat_as}" ]]; then
  while IFS= read -r d; do
    add_candidate "${d}"
    wechat_deep_found=1
  done < <(find "${wechat_as}" -maxdepth 5 -type d -name "MessageTemp" 2>/dev/null)
  [[ "${wechat_deep_found}" == "1" ]] || add_candidate "${wechat_as}"
fi

# QQ / 企业微信 / 钉钉
add_candidate "${HOME}/Library/Containers/com.tencent.qq/Data/Documents"
add_candidate "${HOME}/Library/Containers/com.tencent.WeWorkMac/Data/Library/Application Support/WXWork"
add_candidate "${HOME}/Library/Application Support/DingTalk"
add_candidate "${HOME}/Library/Application Support/iDingTalk"

[[ ${#candidates[@]} -gt 0 ]] || fail "未探测到任何候选目录"

# TCC 可读性检测（ls 失败 ≈ 未授权或权限异常）
readable=()
unreadable_count=0
for dir in "${candidates[@]}"; do
  if ls "${dir}" >/dev/null 2>&1; then
    readable+=("y")
  else
    readable+=("n")
    unreadable_count=$((unreadable_count + 1))
  fi
done

info "探测到 ${#candidates[@]} 个候选目录："
for i in "${!candidates[@]}"; do
  mark="✓"
  [[ "${readable[$i]}" == "n" ]] && mark="⚠️ 不可读(TCC?)"
  printf "  [%2d] %s  %s\n" "$((i + 1))" "${candidates[$i]}" "${mark}"
done

if (( unreadable_count > 0 )); then
  echo
  info "⚠️ 有 ${unreadable_count} 个目录不可读：macOS TCC 拦截了未授权访问。"
  info "   请到「系统设置 → 隐私与安全性 → 完全磁盘访问权限」添加你运行"
  info "   本脚本与后端服务的终端 App，然后重启终端再试。"
fi

if [[ "${1:-}" == "--print" ]]; then
  exit 0
fi

echo
read -r -p "[macos-sources] 选择要纳入索引的编号（空格/逗号分隔，a=全选，q=退出）: " ans
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
info "  1. 重启服务: bash scripts/service.sh restart（或前台 make run）"
info "  2. 首扫在启动时自动执行；也可在 Web 页面点「重建索引」"
info "  3. 后端进程同样受 TCC 约束：用哪个终端/方式起服务，就给哪个 App 授权"
info "  4. iCloud 未下载文件（.icloud 占位符）会被自然跳过，不触发下载"
