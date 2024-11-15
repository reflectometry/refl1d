ROOTDIR = $(shell pwd)

# Check for the presence of bun or npm
ifneq ($(shell which bun),)
  FE_CMD=bun
else ifneq ($(shell which npm),)
  FE_CMD=npm
else
  echo "No frontend build tool found. Please install 'bun' or 'npm'."
  exit 1
endif

# This nifty perl one-liner collects all comments headed by the double "#" symbols next to each target and recycles them as comments
.PHONY: help
help: ## Print this help message
	@perl -nle'print $& if m{^[/a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

.PHONY: clean
clean: ## Delete some cruft from builds/testing/etc.
	rm -f `find . -type f -name '*.py[co]'`
	rm -rf `find . -name __pycache__ -o -name "*.egg-info"` \
		`find . -name 'output-*'` \
		.coverage build dist \
		doc/_build doc/api doc/tutorial \
		.pytest_cache \
		.ruff_cache 

.PHONY: test
test: ## Run pytest and doc tests
	pytest -v
	python check_examples.py --chisq

##############################
### Linting and formatting ###
##############################

.PHONY: lint
lint: lint-backend lint-frontend ## Run all linters

.PHONY: lint-backend
lint-backend: ## Run ruff linting on python code
	@ruff check --fix refl1d/ tests/ setup.py

.PHONY: lint-frontend-check
lint-frontend-check: ## Run bun linting check on javascript code
	cd refl1d/webview/client && \
		$(FE_CMD) run test:lint

.PHONY: lint-frontend
lint-frontend: ## Run bun linting fix on javascript code
	cd refl1d/webview/client && \
		$(FE_CMD) run lint

.PHONY: format
format: format-backend format-frontend ## Run all formatters

.PHONY: format-backend
format-backend: ## Run ruff formatting on python code
	@ruff format refl1d/ tests/ setup.py

.PHONY: format-frontend
format-frontend: ## Run bun formatting on javascript code
	cd refl1d/webview/client && \
		$(FE_CMD) run lint