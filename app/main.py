from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse

from .processor import run_analysis



app = FastAPI(title="Data Analyst Agent API", version="1.0.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}




# Alternative version if you need to keep temp files alive longer
@app.post("/api/")
async def receive_files_delayed_cleanup(
    questions_txt: UploadFile | None = File(None, description="Text file with questions"),
    image_png: UploadFile | None = File(None, description="Optional image file"),
    data_csv: UploadFile | None = File(None, description="Optional CSV dataset"),
) -> JSONResponse:
    """
    Version that doesn't immediately clean up temp files.
    Use this if your process_submission function runs asynchronously
    or if the LangChain executor needs more time to access the files.
    """
    # Validate required questions file
    if questions_txt is None:
        raise HTTPException(status_code=400, detail="Missing 'questions_txt' in multipart form data.")

    saved_paths: Dict[str, str] = {}
    
    try:
        # Save questions.txt to /tmp
        questions_bytes = await questions_txt.read()
        questions_temp = tempfile.NamedTemporaryFile(
            mode='wb', 
            delete=False,  # Don't auto-delete
            suffix='.txt',
            prefix='questions_'
        )
        questions_temp.write(questions_bytes)
        questions_temp.close()
        
        questions_path = Path(questions_temp.name)
        saved_paths["questions_txt"] = str(questions_path.absolute())
        
        # Decode questions text
        try:
            questions_text = questions_bytes.decode("utf-8", errors="replace")
        except Exception:
            questions_text = questions_bytes.decode(errors="replace")

        # Save image if provided
        if image_png is not None:
            image_bytes = await image_png.read()
            image_temp = tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                suffix='.png',
                prefix='image_'
            )
            image_temp.write(image_bytes)
            image_temp.close()
            saved_paths["image_png"] = str(Path(image_temp.name).absolute())

        # Save CSV if provided
        if data_csv is not None:
            csv_bytes = await data_csv.read()
            csv_temp = tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                suffix='.csv',
                prefix='data_'
            )
            csv_temp.write(csv_bytes)
            csv_temp.close()
            saved_paths["data_csv"] = str(Path(csv_temp.name).absolute())

        # Build submission text
        lines: List[str] = [questions_text.rstrip("\n")]
        if "image_png" in saved_paths:
            lines.append(f"image path: {saved_paths['image_png']}")
        if "data_csv" in saved_paths:
            lines.append(f"data.csv path: {saved_paths['data_csv']}")

        submission_text = "\n".join(lines)

        # Process submission
        result = run_analysis(submission_text)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


if __name__ == "__main__":
    # For local development: python -m app.main
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)