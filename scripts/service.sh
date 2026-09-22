#!/usr/bin/env bash
# ============================================================================
# 端云结合 Agent 服务管理脚本
#
# 用法：
#   bash scripts/service.sh start          # 启动
#   bash scripts/service.sh stop           # 停止
#   bash scripts/service.sh restart        # 重启
#   bash scripts/service.sh status         # 运行状态 + 健康检查
#   bash scripts/service.sh logs [-f] [-n N]   # 查看日志（默认 200 行）
#   bash scripts/service.sh health         # 仅健康检查
#
# 两种托管模式（自动选择）：
#   1. systemd  —— /etc/systemd/system/edge-cloud-agent.service 存在时（deploy.sh 安装）
#   2. PID 文件 —— 无 systemd（开发机/容器/无 root）时降级：nohup + .server.pid + logs/
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

UNIT_NAME="edge-cloud-agent"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}.service"
PID_FILE="${ROOT_DIR}/.server.pid"
LOG_FILE="${ROOT_DIR}/logs/edge-cloud-agent.log"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"   # 启动同步加载 ~9.5GiB 模型，给足超时

info() { echo "[service] $*"; }
fail() { echo "[service][ERROR] $*" >&2; exit 1; }

SUDO=""
if [[ ${EUID} -ne 0 ]] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

# 端口解析优先级：进程环境 PORT > systemd unit 内 Environment=PORT > .env PORT > 9000
resolve_port() {
  if [[ -n "${PORT:-}" ]]; then printf '%s' "${PORT}"; return; fi
  if [[ -r "${UNIT_PATH}" ]]; then
    local p
    p="$(sed -nE 's/^Environment=PORT=([0-9]+)[[:space:]]*$/\1/p' "${UNIT_PATH}" | head -n1 || true)"
    if [[ -n "${p}" ]]; then printf '%s' "${p}"; return; fi
  fi
  if [[ -f .env ]]; then
    local v
    v="$(grep -E '^[[:space:]]*PORT=' .env | tail -n1 | cut -d= -f2- | tr -d '\r"'"'" || true)"
    if [[ -n "${v}" ]]; then printf '%s' "${v}"; return; fi
  fi
  printf '9000'
}
PORT="$(resolve_port)"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

use_systemd() {
  [[ -f "${UNIT_PATH}" ]] && command -v systemctl >/dev/null 2>&1
}

pid_alive() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

health_ok() {
  command -v curl >/dev/null 2>&1 \
    && curl -fsS --max-time 5 "${HEALTH_URL}" 2>/dev/null | grep -q '"ok"'
}

wait_health() {
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    health_ok && return 0
    sleep 5
  done
  return 1
}

do_start() {
  if use_systemd; then
    ${SUDO} systemctl start "${UNIT_NAME}.service"
    info "systemd 已拉起 ${UNIT_NAME}，等待就绪（模型加载以分钟计，最长 ${HEALTH_TIMEOUT}s）..."
    if wait_health; then info "健康检查通过: ${HEALTH_URL} ✓"
    else fail "启动超时；查看日志: bash scripts/service.sh logs"; fi
  else
    if pid_alive; then info "已在运行 (pid $(cat "${PID_FILE}"))"; return; fi
    mkdir -p logs
    nohup bash scripts/run.sh >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    info "PID 托管模式已启动 (pid $(cat "${PID_FILE}"))，等待就绪..."
    if wait_health; then info "健康检查通过: ${HEALTH_URL} ✓"
    else
      warn_tail
      fail "启动超时；查看日志: bash scripts/service.sh logs"
    fi
  fi
}

warn_tail() {
  info "最近日志："
  if use_systemd; then
    ${SUDO} journalctl -u "${UNIT_NAME}" -n 30 --no-pager 2>/dev/null || true
  else
    tail -n 30 "${LOG_FILE}" 2>/dev/null || true
  fi
}

do_stop() {
  if use_systemd; then
    ${SUDO} systemctl stop "${UNIT_NAME}.service"
    info "已停止（systemd）"
  else
    if pid_alive; then
      local pid; pid="$(cat "${PID_FILE}")"
      kill "${pid}" 2>/dev/null || true
      # 最多等 30s 优雅退出（watch 线程停止 + 落盘），仍存活则强杀
      for _ in $(seq 1 30); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
      done
      kill -9 "${pid}" 2>/dev/null || true
      rm -f "${PID_FILE}"
      info "已停止 (pid ${pid})"
    else
      rm -f "${PID_FILE}"
      info "未在运行"
    fi
  fi
}

do_status() {
  if use_systemd; then
    ${SUDO} systemctl --no-pager --full status "${UNIT_NAME}.service" 2>/dev/null \
      | head -n 15 || info "unit 未加载"
  elif pid_alive; then
    info "PID 托管模式运行中 (pid $(cat "${PID_FILE}"))"
  else
    info "未在运行"
  fi
  if health_ok; then info "健康检查: ${HEALTH_URL} ✓"
  else info "健康检查: ${HEALTH_URL} ✗（未就绪或未运行）"; fi
}

do_logs() {
  local follow=0 lines=200
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -f|--follow) follow=1 ;;
      -n) lines="${2:-200}"; [[ $# -ge 2 ]] && shift ;;
      *) fail "logs 未知参数: $1（支持 -f / -n N）" ;;
    esac
    shift
  done
  if use_systemd; then
    if [[ "${follow}" == "1" ]]; then
      exec ${SUDO} journalctl -u "${UNIT_NAME}" -n "${lines}" -f
    else
      ${SUDO} journalctl -u "${UNIT_NAME}" -n "${lines}" --no-pager
    fi
  else
    [[ -f "${LOG_FILE}" ]] || fail "无日志文件: ${LOG_FILE}（服务从未以 PID 模式启动过？）"
    if [[ "${follow}" == "1" ]]; then
      exec tail -n "${lines}" -f "${LOG_FILE}"
    else
      tail -n "${lines}" "${LOG_FILE}"
    fi
  fi
}

do_health() {
  if health_ok; then
    info "${HEALTH_URL} ✓"
    curl -fsS --max-time 5 "${HEALTH_URL}" 2>/dev/null || true
    echo
  else
    fail "${HEALTH_URL} ✗"
  fi
}

cmd="${1:-help}"
[[ $# -gt 0 ]] && shift || true
case "${cmd}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
  logs)    do_logs "$@" ;;
  health)  do_health ;;
  *) awk 'NR==1 && /^#!/ {next} /^#( |$)/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}" ;;
esac
