#!/bin/bash
set -e
python3 -W ignore::FutureWarning src/main.py --op $*