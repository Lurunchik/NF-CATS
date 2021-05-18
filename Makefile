CODE = nfcats
VENV = poetry run
WIDTH = 120

.PHONY: pretty lint

pretty:
	$(VENV) black  --skip-string-normalization --line-length $(WIDTH) $(CODE)
	$(VENV) isort --apply --recursive --line-width $(WIDTH) $(CODE)
	$(VENV) unify --in-place --recursive $(CODE)

lint:
	$(VENV) black --check --skip-string-normalization --line-length $(WIDTH) $(CODE)
	$(VENV) flake8 --statistics --max-line-length $(WIDTH) $(CODE)
	$(VENV) pylint --rcfile=setup.cfg $(CODE)
	$(VENV) mypy $(CODE)
