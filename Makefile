.PHONY: help install run

help:
	@echo "Available targets:"
	@echo "  make install    Install dependencies"
	@echo "  make run        Start service"
	@echo "  make help       Show this help"

install:
	pip install -r requirements.txt

run:
	bash scripts/run.sh

lint:
	echo "Add formatter/test checks here when needed."
