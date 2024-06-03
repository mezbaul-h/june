#!/usr/bin/env bash

black --diff --check .
isort --diff --check .
mypy --install-types --non-interactive .
bandit --recursive . --configfile pyproject.toml
find . -iname "*.py" \
  -not -path "./venv/*" \
  -not -path "./build/*" \
  -not -path "./node_modules/*" \
  -not -path "*/migrations/*" | xargs pylint
