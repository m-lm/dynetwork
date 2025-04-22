#!/bin/sh
set -e
python3 -m cProfile -s tottime -o perf.log src/main.py --op $*