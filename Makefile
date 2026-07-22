.PHONY: help install style typecheck push

help:
	@echo "Makefile for finance-agent"
	@echo "Usage:"
	@echo "  make install          Install dependencies"
	@echo "  make style            Lint & Format"
	@echo "  make typecheck        Typecheck"
	@echo "  make push             Push agent to Valkyrie"

PYTHON_VERSION ?= 3.11

install:
	uv venv --python $(PYTHON_VERSION)
	uv sync --dev
	uv tool install .
	@echo "🎉 Done! Run 'source .venv/bin/activate' to activate the environment locally."

venv_check:
	@if [ ! -f .venv/bin/activate ]; then \
		echo "❌ Virtualenv not found! Run \`make install\` first."; \
		exit 1; \
	fi

format: venv_check
	@uv run ruff format .
lint: venv_check
	@uv run ruff check --fix .
style: format lint

typecheck: venv_check
	@uv run basedpyright

push:
	valkyrie agent push . --name docent-finance-agent
