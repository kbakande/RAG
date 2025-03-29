#!/bin/bash

# Install poetry
pip install poetry

# Install dependencies from pyproject.toml
poetry install

# Activate the virtual environment poetry created
# Spaces runs app.py directly, so we make sure deps are ready
