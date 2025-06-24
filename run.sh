#!/usr/bin/env bash
# Auto-run script: clean, build, run & log

set -e

echo "Cleaning previous builds and output..."
make clean

echo "Building project..."
make

echo "Running program and logging to output.txt..."
# Run with default data and processed directories, log both stdout and stderr
./bin/channel_separation data processed 2>&1 | tee output.txt

echo "Done. Check output.txt for program logs."
