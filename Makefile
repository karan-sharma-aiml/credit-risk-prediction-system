.PHONY: install pipeline train test api dashboard seed up lint

install:
	pip install -r requirements.txt

pipeline:
	python -m ml.pipeline

train:
	python -m ml.train

test:
	pytest

api:
	uvicorn backend.app.main:app_factory --factory --reload

dashboard:
	streamlit run dashboard/app.py

seed:
	python -m backend.app.scripts.seed_demo --n 300

up:
	docker compose up --build

lint:
	python -m pyflakes ml utils backend dashboard tests
