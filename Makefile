.PHONY: format lint

all:
	@echo "unimplemented"

format:
	@echo "[Makefile] formatting..."
	@black .
	@echo "[Makefile] sorting imports..."

lint:
	@echo "[Makefile] running static analyzer..."
	@mypy .

fl: format lint
	@format
	@lint
