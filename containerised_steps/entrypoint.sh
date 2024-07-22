#!/bin/bash
set -e
poetry shell

exec "$@"