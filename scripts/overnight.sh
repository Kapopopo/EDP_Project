#!/usr/bin/env bash
# Alias vers le pipeline complet (3 phases automatiques).
exec "$(cd "$(dirname "$0")" && pwd)/go.sh" "$@"
