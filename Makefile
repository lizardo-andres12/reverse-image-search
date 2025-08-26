.PHONY: format lint

all:
	@echo "unimplemented"

format:
	@echo "[Makefile] formatting..."
	@black src/
	@echo "[Makefile] sorting imports..."
	@isort src/

lint:
	@echo "[Makefile] running static analyzer..."
	@mypy src/

fl: format lint
	@format
	@lint
