#!/usr/bin/env bash
~/.pyenv/versions/3.11.9/bin/python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt