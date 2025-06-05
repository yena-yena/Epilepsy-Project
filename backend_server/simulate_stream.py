from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Flutter 앱 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 미리 저장된 EEG 데이터 불러오기
X = np.load("npy/X_dwt.npy")   # shape: (N, 8, 38)
y = np.load("npy/y_total.npy") # shape: (N,)
ids = np.load("npy/id_list.npy")

current_index = {"idx": 0}

class StreamResponse(BaseModel):
    data: list
    label: int

@app.get("/")
def root():
    return {"message": "🧠 EEG stream API is running."}

@app.get("/stream", response_model=StreamResponse)
def stream_sample():
    idx = current_index["idx"]

    if idx >= len(X):
        return {"data": [[0]*38]*8, "label": 0}  # 더 이상 없을 때 빈 값 반환

    sample = X[idx]
    label = int(y[idx])
    current_index["idx"] += 1

    return {"data": sample.tolist(), "label": label}
