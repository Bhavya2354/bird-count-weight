from fastapi import FastAPI, UploadFile, File
import shutil
import os
import json

app = FastAPI(title="Bird Counting & Weight Estimation API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze_video")
def analyze_video(file: UploadFile = File(...)):
    input_path = "data/videos/sample.mp4"

    # Save uploaded video
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run full pipeline
    os.system("python test_tracking.py")

    # Load JSON results
    with open("outputs/json/counts.json") as f:
        data = json.load(f)

    return {
        "fps": data["fps"],
        "frame_skip": data["frame_skip"],
        "counts_sample": data["counts"][:10],
        "output_video": "outputs/videos/tracking_output.mp4"
    }
