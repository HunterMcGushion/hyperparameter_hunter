@ENV=$(ENV)
@PYTHON=$(PYTHON)

FORCE:

env: FORCE
	sh ./scripts/env.sh $(ENV) $(PYTHON)

setup: env
	sh ./scripts/setup.sh $(ENV)

dev: setup
	sh ./scripts/dev.sh $(ENV)

release: lint test clean
	sh ./scripts/release.sh

format:
	black hyperparameter_hunter setup.py
	isort

lint:
	black --check hyperparameter_hunter setup.py

test:
	nosetests

clean:
	rm -rf build dist *.egg-info

distclean: clean
	rm -rf env