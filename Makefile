.PHONY: help install run run-any sources test lint check deploy uninstall start stop restart status logs

# 统一走项目 venv。PATH 上的裸 python3/pytest 可能指向 conda 等其它解释器：
# 实测本机 `make test` 会跑 miniconda 的 Python 3.14 + fastapi 0.139，而
# requirements.txt 钉的是 fastapi 0.116，得到的结果与真实运行环境不一致。
# 解析顺序与 scripts/run.py 一致：.venv -> $$VIRTUAL_ENV，bin/ 或 Scripts/。
PY := $(shell for p in .venv/bin/python .venv/Scripts/python.exe \
	"$$VIRTUAL_ENV/bin/python" "$$VIRTUAL_ENV/Scripts/python.exe"; do \
	if [ -x "$$p" ]; then echo "$$p"; break; fi; done)
PY := $(if $(PY),$(PY),python3)


help:
	@echo "Available targets:"
	@echo "  make install    Install dependencies"
	@echo "  make run        Start service (foreground, dev; bash)"
	@echo "  make run-any    Start service via cross-platform scripts/run.py"
	@echo "                  (Windows 原生无 make 时: python scripts\\run.py)"
	@echo "  make sources    Configure index source dirs (cross-platform, scripts/sources.py)"
	@echo "  make test       Run tests (uses the project venv, not PATH python)"
	@echo "  make lint       Static checks (ruff; config in ruff.toml)"
	@echo "  make check      lint + test"
	@echo "  make deploy     One-shot server deploy (Linux + systemd, see docs/DEPLOYMENT.md)"
	@echo "  make start      Start deployed service"
	@echo "  make stop       Stop service"
	@echo "  make restart    Restart service"
	@echo "  make status     Service status + health check"
	@echo "  make logs       Follow service logs"
	@echo "  make uninstall  Remove systemd unit (keeps code/models/data)"
	@echo "  make help       Show this help"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r requirements-dev.txt

run:
	bash scripts/run.sh

run-any:
	python3 scripts/run.py

sources:
	python3 scripts/sources.py

test:
	$(PY) -m pytest tests/

deploy:
	bash scripts/deploy.sh

uninstall:
	bash scripts/deploy.sh --uninstall

start:
	bash scripts/service.sh start

stop:
	bash scripts/service.sh stop

restart:
	bash scripts/service.sh restart

status:
	bash scripts/service.sh status

logs:
	bash scripts/service.sh logs -f

# 只做静态检查（ruff）。格式化刻意不接：`ruff format` 会重排 27 个文件，
# 纯外观改动应单独成一个提交，不要和功能性修改混在一起。
lint:
	$(PY) -m ruff check src/ tests/ scripts/

check: lint test
