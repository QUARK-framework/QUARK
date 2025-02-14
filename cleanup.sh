#!/bin/bash

echo "Cleaning up old environment..."
rm -f pyproject.toml
rm -f uv.lock
rm -f .python-version
rm -rf .venv
rm -rf .settings/envs