@ENV=$(ENV)
@PYTHON=$(PYTHON)

FORCE:

env: FORCE
	sh ./scripts/env.sh $(ENV) $(PYTHON)

setup: env
	sh ./scripts/setup.sh $(ENV)

dev: setup
	sh ./scripts/dev.sh $(ENV)

ml_install:
	sh ./scripts/ml_install.sh $(ENV)

release: clean lint test
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
	rm -f hyperparameter_hunter/library_helpers/__temp_model_builder.py

distclean: clean
	rm -rf env