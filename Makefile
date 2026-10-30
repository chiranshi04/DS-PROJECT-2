.PHONY: install lint format test audit-data build-manifests train evaluate robustness export app
install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .
lint:
	ruff check .
format:
	ruff format .
test:
	pytest
audit-data:
	python scripts/audit_data.py
build-manifests:
	python scripts/build_manifests.py
train:
	python -m indivoice.train --config configs/experiments/$(or $(EXP),p1_b1_small).yaml
evaluate:
	python scripts/evaluate.py
robustness:
	python scripts/robustness.py
export:
	python scripts/export.py
app:
	uvicorn app.main:app --host 127.0.0.1 --port 8000
