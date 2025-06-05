from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from model import CNNBiLSTMModel

app = FastAPI()

class EEGInput(BaseModel):
    data: list  # 2D list [channels, time]

# 모델 초기화
MODEL_PATH = "saved_models/model_fold1_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBiLSTMModel(input_channels=8, input_time=38).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

@app.get("/")
def root():
    return {"message": "EEG seizure prediction model ready."}

@app.post("/predict")
def predict(input_data: EEGInput):
    try:
        arr = np.array(input_data.data)
        if arr.shape != (8, 38):
            raise ValueError("입력은 [8, 38] 형태여야 합니다.")

        with torch.no_grad():
            tensor_input = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            output = model(tensor_input).squeeze().item()
            prob = torch.sigmoid(torch.tensor(output)).item()
            return {"prediction": int(prob > 0.5), "probability": prob}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
