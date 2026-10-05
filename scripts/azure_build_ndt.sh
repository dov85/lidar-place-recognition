#!/usr/bin/env bash
# Kept for the commands in older notes: same as METHOD=ndt bash scripts/azure_build.sh
METHOD=ndt exec bash "$(dirname "$0")/azure_build.sh" "$@"
