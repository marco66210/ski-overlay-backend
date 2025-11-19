# backend/main.py
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from video_processing import process_videos

# Valeurs par défaut côté API
MAX_DURATION_DEFAULT = 40.0  # secondes max de vidéo à utiliser
DEFAULT_OUTPUT_FPS = 30      # fps de sortie

app = FastAPI()


@app.get("/ping")
def ping():
    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.post("/api/process")
async def process_endpoint(
    video1: UploadFile = File(...),
    video2: UploadFile = File(...),
    max_duration_s: float = Query(MAX_DURATION_DEFAULT, ge=1, le=60),
    output_fps: int = Query(DEFAULT_OUTPUT_FPS, ge=10, le=60),
):
    """
    Traite deux vidéos et renvoie un ID pour récupérer la vidéo superposée.
    """
    if not video1 or not video2:
        raise HTTPException(status_code=400, detail="Deux vidéos sont nécessaires.")

    vid1_id = uuid.uuid4().hex
    vid2_id = uuid.uuid4().hex
    out_id = uuid.uuid4().hex

    p1 = UPLOAD_DIR / f"{vid1_id}.mp4"
    p2 = UPLOAD_DIR / f"{vid2_id}.mp4"
    out_path = OUTPUT_DIR / f"{out_id}.mp4"

    with p1.open("wb") as f:
        shutil.copyfileobj(video1.file, f)
    with p2.open("wb") as f:
        shutil.copyfileobj(video2.file, f)

    try:
        process_videos(
            video1_path=p1,
            video2_path=p2,
            output_path=out_path,
            max_duration_s=max_duration_s,
            output_fps=output_fps,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traitement vidéo : {e}")

    return {"output_id": out_id}


@app.get("/api/output/{output_id}")
def get_output(output_id: str):
    """
    Télécharge la vidéo superposée générée.
    """
    p = OUTPUT_DIR / f"{output_id}.mp4"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Vidéo non trouvée.")
    return FileResponse(str(p), media_type="video/mp4", filename="overlay.mp4")
