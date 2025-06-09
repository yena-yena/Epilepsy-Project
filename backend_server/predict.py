from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
from pydantic import BaseModel
import os
from typing import List

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        print(*[str(a).encode('utf-8', 'ignore').decode('utf-8', 'ignore') for a in args], **kwargs)

safe_print("\U0001f9e0✅ [LOG] FastAPI 서버 실행됨 — 현재 predict.py 최신 버전!")

# ✅ 모델 구조 정의 (Fold1과 일치)
class CNNBiLSTMModel(nn.Module):
    def __init__(self, input_channels, input_time):
        super(CNNBiLSTMModel, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(32 * 2, 1)

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(1)
        if x.shape[1] != self.input_channels:
            x = x.permute(0, 2, 1)  # [B, C, T] → [B, T, C]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # [B, 64, T']
        x = x.permute(0, 2, 1)  # [B, T', 64]
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # 마지막 타임스텝 출력
        x = self.fc(x)
        return x

# ✅ 모델 로드
model = CNNBiLSTMModel(input_channels=8, input_time=80)
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "saved_models/model_fold1_best.pt")

try:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    safe_print("✅ Fold 1 모델 로드 성공!")
except Exception as e:
    safe_print(f"❌ 모델 로드 실패: {e}")

# ✅ FastAPI 앱 초기화
app = FastAPI()

# ✅ CORS 설정 (Flutter 연동 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 입력 데이터 정의
class EEGInput(BaseModel):
    data: List[List[float]]  # shape: [8][80]

@app.post("/predict")
async def predict(input_data: EEGInput):
    try:
        # 1. 입력 받기
        data = np.array(input_data.data)  # shape: (8, 80)
        safe_print("\ud83d\udcc5 입력 shape:", data.shape)
        safe_print("\ud83d\udcc5 입력 일부:\n", data[:, :5])

        # 2. 차원 맞춰 Tensor 변환
        if data.shape != (8, 80):
            return {"error": f"Invalid shape: {data.shape}, expected (8, 80)"}
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # shape: [1, 8, 80]

        # ✅ 추가 디버깅 출력
        safe_print(f"\ud83d\udccc 입력 텐서 평균: {tensor.mean().item():.6f}, 표준편차: {tensor.std().item():.6f}")
        safe_print(f"\U0001f9e0 모델 입력 shape: {tensor.shape}")

        # 3. 추론
        output = model(tensor)
        safe_print(f"\u2699\ufe0f 모델 출력 Tensor: {output}")
        safe_print(f"\u2699\ufe0f 모델 원시 출력값 (item): {output.item()}")

        prob = torch.sigmoid(output).item()
        safe_print(f"\ud83d\udcc8 예측 확률 (sigmoid): {prob:.4f}")

        return {
            "prediction": int(prob > 0.5),
            "probability": prob
        }

    except Exception as e:
        safe_print(f"❌ 예측 오류: {e}")
        return {"error": str(e)}

@app.get("/stream")
def stream_sample():
    dummy_data = np.random.randn(8, 80).tolist()  # ✅ 순서 주의! [8][80]
    return {"data": dummy_data, "label": 0}

# ✅ 수동 테스트 (선택적 실행)
if __name__ == "__main__":
    safe_print("\ud83e\uddea 모델 수동 테스트 시작")
    dummy = np.random.randn(8, 80).astype(np.float32)
    tensor = torch.tensor(dummy).unsqueeze(0)
    output = model(tensor)
    prob = torch.sigmoid(output).item()
    safe_print(f"\ud83e\uddea 수동 예측 확률: {prob:.4f}")
