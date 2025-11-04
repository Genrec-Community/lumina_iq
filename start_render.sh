#!/bin/bash
set -e

echo "Starting Lumina IQ Backend on Render..."

# Set Python path to include the project root
export PYTHONPATH=/opt/render/project/src:$PYTHONPATH

# Change to project directory
cd /opt/render/project/src

# Run gunicorn with the correct module path
uv run gunicorn backend.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
