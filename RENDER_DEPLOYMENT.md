# Render Deployment Guide for Lumina IQ

## Issue Fixed

**Problem:** `gunicorn.errors.AppImportError: Failed to find attribute 'app' in 'main'.`

**Cause:** Render was trying to run `gunicorn main:app` but the FastAPI app is in `backend/main.py`, not root `main.py`.

## Solution

### Option 1: Use render.yaml (Recommended)

The `render.yaml` file is now configured correctly. Render will automatically use it.

### Option 2: Update Start Command in Render Dashboard

Go to your Render service settings and update the **Start Command** to:

```bash
uv run gunicorn backend.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
```

### Option 3: Use the Startup Script

Use the provided `start_render.sh`:

```bash
bash start_render.sh
```

## Environment Variables Required

Make sure these are set in your Render service:

### Required:
- `TOGETHER_API_KEY` - Your Together.ai API key
- `QDRANT_URL` - Your Qdrant cloud URL
- `QDRANT_API_KEY` - Your Qdrant API key
- `PORT` - Auto-set by Render

### Optional (with defaults):
- `ENVIRONMENT=production`
- `HOST=0.0.0.0`
- `QDRANT_COLLECTION_NAME=lumina_iq_documents_prod`
- `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`
- `EMBEDDING_DIMENSIONS=768`
- `LOG_LEVEL=INFO`
- `CORS_ORIGINS=["*"]`

### Auto-generated (Render will create):
- `JWT_SECRET` - Will be auto-generated
- `ENCRYPTION_KEY` - Will be auto-generated

## Build Command

```bash
pip install uv && uv sync
```

## Start Command

```bash
cd /opt/render/project/src/backend && uv run gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
```

## Verify Deployment

Once deployed, test these endpoints:

1. **Health Check:**
   ```bash
   curl https://your-app.onrender.com/
   ```
   Expected: `{"message":"Learning App API is running"}`

2. **API Docs:**
   ```
   https://your-app.onrender.com/docs
   ```

3. **Health Status:**
   ```bash
   curl https://your-app.onrender.com/health
   ```

## Troubleshooting

### Port Not Detected

Make sure you're binding to `0.0.0.0:$PORT` (not just `PORT` or a hardcoded port).

### Import Errors

The command must use `backend.main:app` (not `main:app`).

### Timeout Issues

Increase `--timeout` if startup takes longer:
```bash
--timeout 180
```

### Worker Crashes

Reduce workers if running out of memory:
```bash
--workers 2
```

## Files Added

- `render.yaml` - Render service configuration
- `Procfile` - Alternative process file
- `start_render.sh` - Startup script with environment setup
- `RENDER_DEPLOYMENT.md` - This documentation

## Quick Fix

If you're already in Render dashboard:

1. Go to your service → Settings
2. Update **Start Command** to:
   ```
   cd /opt/render/project/src/backend && uv run gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
   ```
3. Click **Save Changes**
4. Trigger a new deploy

That's it! ✅
