#!/usr/bin/env bash
set -o errexit  # exit if any command fails

# Always upgrade pip, setuptools, and wheel first
pip install --upgrade pip setuptools wheel

# Then install your dependencies
pip install -r requirements.txt
