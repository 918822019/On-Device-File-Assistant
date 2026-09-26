@echo off
rem Windows 原生启动入口：薄包装转发到跨平台的 run.py
rem 用法: scripts\run.bat   (等价于 python scripts\run.py)
where python >nul 2>nul
if errorlevel 1 (
  echo [run.bat][ERROR] 找不到 python，请先安装 Python 3.11+ 并加入 PATH >&2
  exit /b 1
)
python "%~dp0run.py" %*
exit /b %errorlevel%
