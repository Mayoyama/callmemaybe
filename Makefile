.PHONY: install run debug clean lint lint-strict uninstall

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src/__main__.py

lint:
	uv run python -m flake8 . --exclude=.venv,llm_sdk/
	uv run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude=.venv,llm_sdk/

lint-strict:
	uv run python -m flake8 . --exclude=.venv,llm_sdk/
	uv run python -m mypy . --strict --exclude=.venv,llm_sdk/

clean:
	find . -not -path './$(VENV)/*' -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*~" \) -delete

uninstall: clean
	rm -rf .venv