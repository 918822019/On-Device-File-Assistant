#!/usr/bin/env bash
# ============================================================================
# 端云结合 Agent 一键部署脚本（Linux 服务器 + systemd）
#
# 流程：前置检查 → venv/依赖 → .env 校验生成 → 模型权重检查/下载
#       → systemd unit 安装 → 启动 → 健康检查
#
# 用法：
#   bash scripts/deploy.sh                 # 交互式一键部署
#   bash scripts/deploy.sh --yes           # 非交互（CI/自动化），大体积下载不再确认
#   bash scripts/deploy.sh --skip-models   # 权重已就位时跳过检查/下载
#   bash scripts/deploy.sh --skip-deps     # 跳过 venv/依赖安装
#   bash scripts/deploy.sh --skip-systemd  # 不装 systemd unit（无 root / 容器内）
#   bash scripts/deploy.sh --no-start      # 装好但不启动
#   bash scripts/deploy.sh --gpu           # 服务器有 NVIDIA GPU 时，torch 装默认(CUDA)轮子
#   bash scripts/deploy.sh --port 8080     # 自定义端口（默认 9000）
#   bash scripts/deploy.sh --uninstall     # 卸载 systemd 服务（不动代码/权重/数据）
#
# 说明：请以普通用户运行；仅 systemd 安装步骤会自动通过 sudo 提权。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

UNIT_NAME="edge-cloud-agent"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}.service"
UNIT_TEMPLATE="${ROOT_DIR}/deploy/${UNIT_NAME}.service"

# ---------------------------- 默认参数 ----------------------------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"
ASSUME_YES=0
SKIP_DEPS=0
SKIP_MODELS=0
SKIP_SYSTEMD=0
NO_START=0
GPU=0
UNINSTALL=0
PYTHON_BIN=""
# 启动阶段同步加载 ~9.5GiB 端侧模型（CPU 上以分钟计），健康检查须给足超时
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
# 权重下载前的磁盘余量门槛（GiB）：LLM ~9.6 + embedding ~1.1 + 解压/缓存余量
MIN_DISK_GIB="${MIN_DISK_GIB:-15}"
# 常驻内存门槛（GiB）：E2B bfloat16 权重 ~9.5GiB + 运行时开销
MIN_MEM_GIB="${MIN_MEM_GIB:-12}"

usage() {
  # 只打印文件头部的注释块（遇第一行非注释即停），不扫全文
  awk 'NR==1 && /^#!/ {next} /^#( |$)/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)       ASSUME_YES=1 ;;
    --skip-deps)    SKIP_DEPS=1 ;;
    --skip-models)  SKIP_MODELS=1 ;;
    --skip-systemd) SKIP_SYSTEMD=1 ;;
    --no-start)     NO_START=1 ;;
    --gpu)          GPU=1 ;;
    --uninstall)    UNINSTALL=1 ;;
    --host)         HOST="$2"; shift ;;
    --port)         PORT="$2"; shift ;;
    --python)       PYTHON_BIN="$2"; shift ;;
    -h|--help)      usage ;;
    *) echo "[deploy] 未知参数: $1（-h 查看用法）" >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------- 基础工具 ----------------------------
info()  { echo "[deploy] $*"; }
warn()  { echo "[deploy][WARN] $*" >&2; }
fail()  { echo "[deploy][ERROR] $*" >&2; exit 1; }

SUDO=""
if [[ ${EUID} -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
fi

RUN_USER="${SUDO_USER:-$(id -un)}"
if [[ ${EUID} -eq 0 && -n "${SUDO_USER:-}" ]]; then
  RUN_USER="${SUDO_USER}"
fi

confirm() {
  # $1 = 提示语。--yes 直接过；非交互且未 --yes 则拒绝（防止自动化场景误拉 10GB 权重）
  local msg="$1"
  [[ "${ASSUME_YES}" == "1" ]] && return 0
  if [[ ! -t 0 ]]; then
    warn "${msg} —— 非交互环境未加 --yes，已跳过该步骤"
    return 1
  fi
  local ans
  read -r -p "[deploy] ${msg} [y/N] " ans
  [[ "${ans}" =~ ^[Yy] ]]
}

# 读配置：进程环境变量优先，其次 .env 中最后一次出现的 KEY=（与 run.sh 语义一致），
# 最后回退默认值。去除 \r 与成对的首尾引号。
env_get() {
  local key="$1" default="${2:-}"
  if [[ -n "${!key:-}" ]]; then printf '%s' "${!key}"; return; fi
  local line=""
  if [[ -f .env ]]; then
    line="$(grep -E "^[[:space:]]*${key}=" .env | tail -n1 || true)"
  fi
  if [[ -z "${line}" ]]; then printf '%s' "${default}"; return; fi
  local val="${line#*=}"
  val="${val%$'\r'}"
  val="${val%\"}"; val="${val#\"}"
  val="${val%\'}"; val="${val#\'}"
  printf '%s' "${val}"
}

# 写配置：只替换「生效行」（不动注释掉的同名键，避免破坏 .env 内的说明文档）；
# 无生效行则追加——run.sh 取首个生效值、python-dotenv 取末个，追加在两种语义下均胜出。
# 值限定为普通路径/单词（不含 | & 换行），满足本项目全部配置项。
env_set() {
  local key="$1" value="$2"
  touch .env
  if grep -qE "^[[:space:]]*${key}=" .env; then
    sed -i.bak -E "s|^([[:space:]]*)${key}=.*|\1${key}=${value}|" .env
    rm -f .env.bak
  else
    printf '%s=%s\n' "${key}" "${value}" >> .env
  fi
  info ".env: ${key}=${value}"
}

# ---------------------------- 卸载分支 ----------------------------
if [[ "${UNINSTALL}" == "1" ]]; then
  if [[ -f "${UNIT_PATH}" ]]; then
    ${SUDO} systemctl disable --now "${UNIT_NAME}.service" 2>/dev/null || true
    ${SUDO} rm -f "${UNIT_PATH}"
    ${SUDO} systemctl daemon-reload
    info "已卸载 ${UNIT_NAME}（代码、权重、data/ 数据均未删除）"
  else
    info "未安装 systemd unit，无需卸载"
  fi
  exit 0
fi

echo "============================================================"
echo " 端云结合 Agent 部署"
echo "   仓库: ${ROOT_DIR}"
echo "   监听: http://${HOST}:${PORT}"
echo "   用户: ${RUN_USER}"
echo "============================================================"

# ---------------------------- 1. 前置检查 ----------------------------
info "[1/6] 前置检查"

if [[ "${SKIP_SYSTEMD}" != "1" && "${OSTYPE}" != linux* ]]; then
  warn "非 Linux 系统，systemd 步骤将自动跳过（可用 --skip-systemd 消除本提示）"
  SKIP_SYSTEMD=1
fi

# 内存：E2B bfloat16 常驻 ~9.5GiB，低于门槛仅告警不阻断（可能配置了 swap 或纯 embedding 场景）
if [[ -r /proc/meminfo ]]; then
  mem_gib=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
  if (( mem_gib < MIN_MEM_GIB )); then
    warn "物理内存 ${mem_gib}GiB < 建议 ${MIN_MEM_GIB}GiB：加载 E2B 权重（~9.5GiB）可能触发 OOM"
  else
    info "内存 ${mem_gib}GiB ✓"
  fi
fi

# 磁盘：models/ 所在分区余量（df -Pk 为 POSIX 选项，GNU/BSD 通用；单位 1024 字节）
disk_gib=$(( $(df -Pk "${ROOT_DIR}" | awk 'NR==2{print $4}') / 1024 / 1024 ))
if (( disk_gib < MIN_DISK_GIB )); then
  warn "磁盘余量 ${disk_gib}GiB < 建议 ${MIN_DISK_GIB}GiB：权重下载可能失败（--skip-models 可跳过下载）"
else
  info "磁盘余量 ${disk_gib}GiB ✓"
fi

command -v curl >/dev/null 2>&1 || warn "未找到 curl，健康检查将不可用（建议安装）"

# Python 解释器：--python 指定 > 已有 .venv > python3.12/3.11/3.10/3（要求 ≥3.10,<3.14）
pick_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    [[ -x "${PYTHON_BIN}" ]] || fail "--python 指定的解释器不存在: ${PYTHON_BIN}"
    printf '%s' "${PYTHON_BIN}"; return
  fi
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    printf '%s' "${ROOT_DIR}/.venv/bin/python"; return
  fi
  local cand
  for cand in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "${cand}" >/dev/null 2>&1 \
      && "${cand}" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,14) else 1)' 2>/dev/null; then
      printf '%s' "$(command -v "${cand}")"; return
    fi
  done
  fail "未找到 Python 3.10–3.13，请先安装（如 dnf/apt install python3.11）或用 --python 指定"
}
BASE_PYTHON="$(pick_python)"
info "基础解释器: ${BASE_PYTHON} ($("${BASE_PYTHON}" -V 2>&1))"

if [[ ${EUID} -eq 0 && -z "${SUDO_USER:-}" ]]; then
  warn "正以 root 直接运行：venv 与权重将归 root 所有，建议改用普通用户执行本脚本"
fi

# ---------------------------- 2. venv 与依赖 ----------------------------
VENV_PY="${ROOT_DIR}/.venv/bin/python"
if [[ "${SKIP_DEPS}" == "1" ]]; then
  info "[2/6] 跳过依赖安装 (--skip-deps)"
  [[ -x "${VENV_PY}" ]] || fail "--skip-deps 需要 .venv 已存在，请先完整跑一次部署"
else
  info "[2/6] venv 与依赖安装"
  if [[ ! -x "${VENV_PY}" ]]; then
    "${BASE_PYTHON}" -m venv .venv || fail "创建 venv 失败（缺 python3-venv？尝试 dnf/apt install python3.11-venv）"
  fi
  "${VENV_PY}" -m pip install --upgrade pip >/dev/null

  # 无 GPU 的 Linux 服务器：先装 CPU 版 torch，避免默认索引拉数 GB 的 CUDA 轮子。
  # requirements.txt 钉死 torch==2.6.0，预装后后续 pip 视为已满足，不会重复下载。
  if [[ "${OSTYPE}" == linux* && "${GPU}" != "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    info "未检测到 NVIDIA GPU，预装 CPU 版 torch==2.6.0（--gpu 可关闭此行为）"
    "${VENV_PY}" -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
  fi

  "${VENV_PY}" -m pip install -r requirements.txt
fi

mkdir -p data logs

# ---------------------------- 3. .env 校验 ----------------------------
info "[3/6] .env 配置"
if [[ ! -f .env ]]; then
  cp .env.example .env
  info "已从 .env.example 生成 .env（默认：端侧 only、CLOUD_ENABLED=false、DEVICE=cpu）"
fi

# 关键项体检（只告警不阻断，值以 .env.example 注释为权威）
if [[ "$(env_get CLOUD_ENABLED false)" == "true" && -z "$(env_get CLOUD_API_BASE)" ]]; then
  warn "CLOUD_ENABLED=true 但 CLOUD_API_BASE 为空，云端调用将全部失败（有端侧回退，不致命）"
fi
if [[ "$(env_get EDGE_DEVICE auto)" != "cpu" ]] && [[ "${OSTYPE}" != linux* ]]; then
  warn "EDGE_DEVICE=$(env_get EDGE_DEVICE auto)：非 cpu 取值请自行确认平台兼容性（Apple Silicon 必须 cpu）"
fi

# ---------------------------- 4. 模型权重 ----------------------------
weights_present() {
  local dir="$1"
  [[ -n "${dir}" && -f "${dir}/config.json" ]] || return 1
  [[ -n "$(find "${dir}" -maxdepth 2 -name '*.safetensors' -print -quit 2>/dev/null)" ]]
}

# 用 venv 内 python 调 huggingface_hub / modelscope 下载（CLI 名称跨版本不稳，API 稳定）。
# HF_ENDPOINT 环境变量（如 https://hf-mirror.com）会被 huggingface_hub 自动识别，可直接透传。
download_model() {
  local repo="$1" target="$2" kind="$3"
  mkdir -p "${target}"
  if [[ "${kind}" == "modelscope" ]]; then
    "${VENV_PY}" - "${repo}" "${target}" <<'PYEOF'
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PYEOF
  else
    "${VENV_PY}" - "${repo}" "${target}" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=4)
PYEOF
  fi
}

# 确保一个模型就位。$1=用途描述 $2=repo id $3=来源(hf/modelscope) $4=本地目录(可空→用默认) $5=默认目录 $6=.env 键(可空)
ensure_model() {
  local label="$1" repo="$2" kind="$3" local_dir="$4" default_dir="$5" env_key="${6:-}"
  local target="${local_dir:-${default_dir}}"

  if weights_present "${target}"; then
    info "${label}: 权重已就位 ${target}"
  else
    info "${label}: ${target} 无完整权重（需 config.json + *.safetensors）"
    if confirm "从 ${kind} 下载 ${repo}（可能数 GiB）到 ${target}？"; then
      download_model "${repo}" "${target}" "${kind}" \
        || fail "${label} 下载失败；国内网络可试 HF_ENDPOINT=https://hf-mirror.com 或改走 modelscope"
      weights_present "${target}" || fail "${label} 下载后仍未发现权重文件，请检查 ${target}"
      info "${label}: 下载完成 ✓"
    else
      warn "${label}: 跳过下载。服务可启动，但该能力不可用（业务层会自动降级）"
    fi
  fi

  # 原本未显式指向本地目录时回填 .env，保证服务与本脚本看到同一份权重
  if [[ -n "${env_key}" && -z "${local_dir}" ]]; then
    env_set "${env_key}" "${target}"
  fi
}

if [[ "${SKIP_MODELS}" == "1" ]]; then
  info "[4/6] 跳过模型权重检查 (--skip-models)"
else
  info "[4/6] 模型权重检查/下载"
  ensure_model "端侧 LLM" \
    "$(env_get EDGE_MODEL_ID google/gemma-4-E2B-it)" \
    "$(env_get EDGE_MODEL_SOURCE hf)" \
    "$(env_get EDGE_LOCAL_DIR)" \
    "models/google/gemma-4-E2B-it" \
    "EDGE_LOCAL_DIR"
  ensure_model "端侧 embedding" \
    "$(env_get EDGE_EMBEDDING_MODEL_ID google/embeddinggemma-300m)" \
    "$(env_get EDGE_EMBEDDING_SOURCE modelscope)" \
    "$(env_get EDGE_EMBEDDING_LOCAL_DIR)" \
    "models/google/embeddinggemma-300m" \
    "EDGE_EMBEDDING_LOCAL_DIR"
  # 指向本地目录后，source 须离开 modelscope（否则启动时会拿本地路径当模型 ID
  # 先下载一轮再回退，多一条误导性 warning —— 见 .env.example 注释）
  if [[ -n "$(env_get EDGE_EMBEDDING_LOCAL_DIR)" \
        && "$(env_get EDGE_EMBEDDING_SOURCE)" == "modelscope" ]]; then
    env_set "EDGE_EMBEDDING_SOURCE" "local"
  fi
fi

# ---------------------------- 5. systemd 安装 ----------------------------
# SERVICE_STARTED=1 仅当本脚本真正把服务拉起时置位，第 6 步据此决定是否健康检查
# （否则 --skip-systemd/无 sudo 场景会对着没起的端口白等 HEALTH_TIMEOUT 秒）
SERVICE_STARTED=0
if [[ "${SKIP_SYSTEMD}" == "1" ]]; then
  info "[5/6] 跳过 systemd 安装 (--skip-systemd)"
  if [[ "${NO_START}" != "1" ]]; then
    info "无 systemd 模式可手动起服务: bash scripts/service.sh start（PID 文件托管）"
  fi
else
  info "[5/6] systemd unit 安装"
  if [[ ${EUID} -ne 0 && -z "${SUDO}" ]]; then
    warn "无 root 且无 sudo，跳过 systemd 安装；可重跑加 --skip-systemd 消除本提示"
  elif [[ ! -f "${UNIT_TEMPLATE}" ]]; then
    fail "unit 模板缺失: ${UNIT_TEMPLATE}"
  else
    if command -v systemctl >/dev/null 2>&1; then
      unit_tmp="$(mktemp)"
      sed -e "s|__ROOT_DIR__|${ROOT_DIR}|g" \
          -e "s|__RUN_USER__|${RUN_USER}|g" \
          -e "s|__HOST__|${HOST}|g" \
          -e "s|__PORT__|${PORT}|g" \
          "${UNIT_TEMPLATE}" > "${unit_tmp}"
      ${SUDO} cp "${unit_tmp}" "${UNIT_PATH}"
      rm -f "${unit_tmp}"
      ${SUDO} systemctl daemon-reload
      ${SUDO} systemctl enable "${UNIT_NAME}.service" >/dev/null 2>&1 || true
      info "unit 已安装并设为开机自启: ${UNIT_PATH}"
      if [[ "${NO_START}" == "1" ]]; then
        info "--no-start：未启动。之后可 bash scripts/service.sh start"
      else
        info "启动 ${UNIT_NAME} ..."
        ${SUDO} systemctl restart "${UNIT_NAME}.service"
        SERVICE_STARTED=1
      fi
    else
      warn "系统无 systemctl，跳过 unit 安装"
      SKIP_SYSTEMD=1
    fi
  fi
fi

# ---------------------------- 6. 健康检查 ----------------------------
if [[ "${SERVICE_STARTED}" != "1" ]]; then
  info "[6/6] 跳过健康检查（本脚本未启动服务）"
  info "启动并验证: bash scripts/service.sh start"
else
  info "[6/6] 健康检查（启动需同步加载端侧模型，最长等待 ${HEALTH_TIMEOUT}s）"
  url="http://127.0.0.1:${PORT}/health"
  if command -v curl >/dev/null 2>&1; then
    deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    ok=0
    while (( $(date +%s) < deadline )); do
      if curl -fsS --max-time 5 "${url}" 2>/dev/null | grep -q '"ok"'; then
        ok=1; break
      fi
      sleep 5
    done
    if [[ "${ok}" == "1" ]]; then
      info "健康检查通过: ${url} ✓"
    else
      warn "健康检查超时。最近日志："
      if [[ "${SKIP_SYSTEMD}" != "1" ]] && command -v journalctl >/dev/null 2>&1; then
        ${SUDO} journalctl -u "${UNIT_NAME}" -n 30 --no-pager 2>/dev/null || true
      else
        tail -n 30 logs/edge-cloud-agent.log 2>/dev/null || true
      fi
      fail "服务未在 ${HEALTH_TIMEOUT}s 内就绪；排查见 docs/DEPLOYMENT.md「常见问题」"
    fi
  else
    warn "无 curl，跳过健康检查；请手动验证: ${url}"
  fi
fi

echo "============================================================"
echo " 部署完成 ✓"
echo "   健康检查 : curl http://127.0.0.1:${PORT}/health"
echo "   服务管理 : bash scripts/service.sh {start|stop|restart|status|logs}"
echo "   运行日志 : journalctl -u ${UNIT_NAME} -f   (或 make logs)"
echo "   卸载服务 : bash scripts/deploy.sh --uninstall"
echo "============================================================"
