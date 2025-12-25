🐔 Bird Counting & Weight Estimation from Poultry CCTV

Candidate: Bhavya
Role Applied: Machine Learning / AI Engineer Intern

1. Project Overview

This project processes a poultry farm CCTV video and automatically produces:

Bird count over time

Stable tracking IDs for each bird

A visual weight proxy

An annotated output video

A JSON file with time-series results

A FastAPI service for inference

This simulates how real poultry farms monitor flock size, movement, and bird growth.

2. What This GitHub Repository Contains

This repository contains everything required to run, understand, and verify the system:

Complete source code

Detection + tracking pipeline

FastAPI backend

Setup & execution instructions

Because GitHub does not allow large ML files, model, videos, and outputs are hosted on Google Drive.

3. Required Files (Google Drive)

Download the following from Google Drive and place them in the project as shown.

Content	Download Link
YOLO poultry model (best.pt)	https://drive.google.com/drive/folders/1HfqeeEwd9XPt83qeKilN_4_S2yQWLHSa?usp=sharing

Input CCTV video (sample.mp4)	https://drive.google.com/drive/folders/1O6RGcFN1Oe_PKI1kUatcVGyVHgytAS1S?usp=sharing


Output video + JSON	https://drive.google.com/drive/folders/1AAmCB-M3ghFMgu1JaG7qZmKL4M2HAKvB?usp=sharing

Place them inside the project like this:

bird_count_weight/
├── models/
│   └── best.pt
├── data/
│   └── videos/
│       └── sample.mp4
├── outputs/
│   ├── videos/
│   │   └── tracking_output.mp4
│   └── json/
│       └── counts.json

4. Folder Structure
bird_count_weight/
├── models/                 ← YOLO model (from Drive)
├── data/
│   └── videos/             ← CCTV video (from Drive)
├── outputs/                ← Results (from Drive)
│   ├── videos/
│   │   └── tracking_output.mp4
│   └── json/
│       └── counts.json
├── app/
│   └── main.py             ← FastAPI server
├── test_tracking.py        ← Full ML pipeline
├── requirements.txt
└── README.md

5. How Detection & Tracking Work

A poultry-trained YOLOv8 model (best.pt) is used.

YOLOv8 performs:

Object detection → finds chickens

ByteTrack tracking → assigns stable IDs

ByteTrack ensures:

Each bird keeps the same ID across frames

Birds are not double-counted

6. How Bird Count Is Computed

For each processed frame:

Bird Count = number of unique active tracking IDs


This creates a time-series:

time → bird count

7. How Weight Is Estimated

True bird weight is unavailable in video, so a visual proxy is used:

Weight Index = bounding box area / frame area


This gives a normalized size estimate of birds.
The system computes the average weight index per frame.

8. How to Run the Project
Step 1 — Install dependencies
pip install -r requirements.txt

Step 2 — Run the ML pipeline
python test_tracking.py


This generates:

outputs/videos/tracking_output.mp4
outputs/json/counts.json

Step 3 — View results

Open:

outputs/videos/tracking_output.mp4


You will see:

Bounding boxes on birds

Stable tracking IDs

Bird count

Weight index

Open:

outputs/json/counts.json


This contains the full time-series.

9. Example JSON Output
{
  "fps": 19.896,
  "frame_skip": 3,
  "counts": [
    {
      "time_sec": 0.15,
      "count": 2,
      "avg_weight_index": 0.0463
    },
    {
      "time_sec": 0.30,
      "count": 2,
      "avg_weight_index": 0.0461
    }
  ]
}

10. Running the API

Start the API:

uvicorn app.main:app --reload


Health check:

http://127.0.0.1:8000/health


API UI:

http://127.0.0.1:8000/docs


Upload sample.mp4 in /analyze_video and click Execute.

Sample API response:

{
  "fps": 19.896,
  "frame_skip": 3,
  "counts_sample": [
    {
      "time_sec": 0.15,
      "count": 2,
      "avg_weight_index": 0.0463
    }
  ],
  "output_video": "outputs/videos/tracking_output.mp4"
}

11. What the Evaluator Should Check

To verify this submission:

Open tracking_output.mp4

Open counts.json

Run the API via /docs

These confirm:

Detection

Tracking

Counting

Weight estimation

12. Final Summary

This system provides:

Poultry-specific YOLO detection

ByteTrack multi-object tracking

Time-based bird counting

Visual weight proxy estimation

API-based inference

It is a complete, production-style poultry CCTV analytics pipeline.