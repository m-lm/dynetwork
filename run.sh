#!/bin/sh
set -e
python3 -W ignore::FutureWarning src/main.py --op $*