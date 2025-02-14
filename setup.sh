#!/bin/bash
PYTHON_VERSION="3.12"

bash ./cleanup.sh

echo "Initializing UV..."
uv init --python=$PYTHON_VERSION
rm -f hello.py

echo "Adding base dependencies..."
uv add inquirer==3.4.0 pyyaml==6.0.2 packaging==24.2

echo "Configuring environment..."
uv run python src/main.py env --configure myenv

if [ -f .settings/envs/requirements_myenv.txt ]; then
  echo "Adding environment-specific dependencies..."
  uv add -r .settings/envs/requirements_myenv.txt
else
  echo "Warning: requirements_myenv.txt not found. Skipping environment-specific dependencies."
fi

echo "Synchronizing dependencies..."
uv sync

echo "Setup complete!"