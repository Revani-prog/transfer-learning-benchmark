from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="Transfer Learning Benchmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load results
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../results/metrics/results.json")

def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    return load_results()