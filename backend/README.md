# Backend

FastAPI API, auth, quotas, audit, and chat/session persistence.

The backend imports shared engine code through the `app.*` namespace. Run commands from the repo root so `sitecustomize.py` adds both `backend/` and `engine/` to `PYTHONPATH`.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```
