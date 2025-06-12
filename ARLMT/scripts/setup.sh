#!/bin/bash
# Development environment setup script

set -e

echo "Setting up ARLMT development environment..."

# Create virtual environment
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov black isort flake8 mypy pre-commit jupyter

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
mkdir -p {models,data,logs,outputs,checkpoints}

echo "Development environment setup completed!"
echo "Activate the environment with: source venv/bin/activate"
