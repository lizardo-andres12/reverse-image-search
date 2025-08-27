.PHONY: format lint

all:
	@echo "unimplemented"

run: src/api/main.py
	@uvicorn main:app --reload --app-dir src/api

dc-up: docker/.env docker/docker-compose.yml
	@docker-compose --env-file docker/.env -f docker/docker-compose.yml up -d

dc-up-debug: docker/.env docker/docker-compose.yml
	@docker-compose --env-file docker/.env -f docker/docker-compose.yml up

dc-down: docker/docker-compose.yml
	@docker-compose -f docker/docker-compose.yml down

clean-mounts: data/
	@sudo rm -rf data/*

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
