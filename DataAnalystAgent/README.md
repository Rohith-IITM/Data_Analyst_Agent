## Data Analyst Agent API

FastAPI app with a health check and a multipart endpoint that accepts files, stores them per request, and forwards a composed text to a pluggable processor function.

### Install

1. Create a virtual environment (optional but recommended):
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Run

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```powershell
curl http://localhost:8000/health
```

### Upload Endpoint

POST `http://localhost:8000/api/` using multipart form:

```powershell
curl -X POST http://localhost:8000/api/ \
  -F "questions.txt=@question.txt" \
  -F "image.png=@image.png" \
  -F "data.csv=@data.csv"
```

Notes:
- Field names should be exactly: `questions.txt`, `image.png`, `data.csv`.
- `questions.txt` is required. Image and CSV are optional.
- Files are saved under `uploads/<timestamp>/`.
- The processor receives a single text value:
  - The full content of `questions.txt`
  - Optionally: `image path: <abs path>`
  - Optionally: `data.csv path: <abs path>`

### Replace the Processor

Edit `app/processor.py` and replace `process_submission` with your own implementation.

