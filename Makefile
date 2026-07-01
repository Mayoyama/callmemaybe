.PHONY: install run run-nested run-50 bonus-encoder verbose debug pytest clean lint uninstall

install:
	uv sync

run:
	uv run python -m src

run-nested:
	uv run python -m src \
      --functions_definition data/input/functions_definition_nested.json \
      --input data/input/function_calling_tests_nested.json \
      --output data/output/results_nested.json

run-50:
	uv run python -m src \
      --functions_definition data/input/functions_definition50.json \
      --input data/input/function_calling_tests50.json \
      --output data/output/results50.json

bonus-encoder:
	uv run python -m src --bonus-encoder

verbose:
	uv run python -m src --verbose

debug:
	uv run python -m pdb -m src

lint:
	uv run python -m flake8 . --exclude=.venv,llm_sdk/
	uv run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude '(\.venv|llm_sdk)'

pytest:
	uv run pytest
clean:
	find . -not -path './.venv/*' -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" \) -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*~" \) -delete

uninstall: clean
	rm -rf .venv