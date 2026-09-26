# Windows 原生启动入口（PowerShell）：薄包装转发到跨平台的 run.py
# 用法: powershell -ExecutionPolicy Bypass -File scripts\run.ps1
$ErrorActionPreference = "Stop"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  Write-Error "[run.ps1][ERROR] 找不到 python，请先安装 Python 3.11+ 并加入 PATH"
  exit 1
}
& python (Join-Path $PSScriptRoot "run.py") @args
exit $LASTEXITCODE
