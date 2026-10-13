#!/bin/zsh
cd "$(dirname "$0")"
uv sync -q
uv run jupyter lab notebooks/octreecity_quickstart.ipynb
