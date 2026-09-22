.PHONY: help install run test deploy uninstall start stop restart status logs lint

help:
	@echo "Available targets:"
	@echo "  make install    Install dependencies"
	@echo "  make run        Start service (foreground, dev)"
	@echo "  make test       Run business-layer tests"
	@echo "  make deploy     One-shot server deploy (Linux + systemd, see docs/DEPLOYMENT.md)"
	@echo "  make start      Start deployed service"
	@echo "  make stop       Stop service"
	@echo "  make restart    Restart service"
	@echo "  make status     Service status + health check"
	@echo "  make logs       Follow service logs"
	@echo "  make uninstall  Remove systemd unit (keeps code/models/data)"
	@echo "  make help       Show this help"

install:
	pip install -r requirements.txt

run:
	bash scripts/run.sh

test:
	pytest tests/

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

lint:
	echo "Add formatter/test checks here when needed."
