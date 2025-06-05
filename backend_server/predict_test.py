from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class EEGInput(BaseModel):
    data: List[List[float]]  # [채널][시간]

class CNNBiLSTMModel(nn.Module):
    def __init__(self, input_channels, input_time):
        super(CNNBiLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(32 * 2, 1)

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(1)
        if x.shape[1] != 8:
            x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)

model = CNNBiLSTMModel(8, 38)
model.load_state_dict(torch.load("saved_models/model_fold1_best.pt", map_location="cpu"))
model.eval()

@app.post("/predict")
async def predict(input_data: EEGInput):
    data = np.array(input_data.data).astype(np.float32)  # [8][38]
    data_tensor = torch.tensor(data).unsqueeze(0)  # [1, 8, 38]
    with torch.no_grad():
        output = torch.sigmoid(model(data_tensor)).item()
    return {
        "prediction": 1 if output > 0.5 else 0,
        "probability": output
    }
