.PHONY: lint format

lint:
	ruff check src/ --line-length 120
	flake8 src/ --max-line-length 120

format:
	ruff format src/
