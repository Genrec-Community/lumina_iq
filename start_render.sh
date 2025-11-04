#!/bin/bash
set -e

echo "Starting Lumina IQ Backend on Render..."

# Change to backend directory so imports work correctly
cd /opt/render/project/src/backend

# Set Python path to include backend directory
export PYTHONPATH=/opt/render/project/src/backend:$PYTHONPATH

# Run gunicorn from backend directory
uv run gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
