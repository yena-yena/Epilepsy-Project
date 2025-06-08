from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List  # ✅ Python 3.8 지원을 위한 typing import
import numpy as np
import torch
from backend_server.model import CNNBiLSTMModel

app = FastAPI()

# ✅ 정확한 데이터 타입 지정 (Python 3.8 호환)
class EEGInput(BaseModel):
    data: List[List[float]]  # Python 3.8 이상에서 호환

# 모델 경로 및 장치 설정
MODEL_PATH = "backend_server/saved_models/model_fold1_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBiLSTMModel(input_channels=8, input_time=38).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

@app.get("/")
def root():
    return {"message": "EEG seizure prediction model ready."}

@app.post("/predict")
async def predict(input_data: EEGInput, request: Request):
    try:
        arr = np.array(input_data.data)
        print("🔥 들어온 EEG 데이터 shape:", arr.shape)

        if arr.shape != (8, 38):
            raise ValueError(f"❌ 입력 shape 오류: {arr.shape} (기대값: (8, 38))")

        with torch.no_grad():
            tensor_input = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            output = model(tensor_input).squeeze().item()
            prob = torch.sigmoid(torch.tensor(output)).item()

            print(f"✅ 예측 완료 - 확률: {prob:.4f}")
            return {
                "prediction": int(prob > 0.5),
                "probability": prob
            }

    except Exception as e:
        body = await request.body()
        print("❌ 예외 발생:", e)
        print("📦 요청 본문:", body.decode("utf-8"))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
