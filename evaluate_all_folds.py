import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import os
import pandas as pd

# ✅ 모델 구조 동일하게 정의
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# ✅ 평가할 모델 리스트
folds = [1, 2, 3, 4]
results = []

# ✅ 테스트셋 불러오기
X_test = np.load("npy/X_test.npy")  # shape: [N, 8, 80]
y_test = np.load("npy/y_test.npy")  # shape: [N]

# 텐서 변환
X_tensor = torch.tensor(X_test, dtype=torch.float32)
y_true = y_test

# ✅ 각 모델 평가
for fold in folds:
    model = CNNLSTM()
    path = f"saved_models/model_fold{fold}_best.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs).squeeze().numpy()
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_true, preds)
        auc = roc_auc_score(y_true, probs)
        f1 = f1_score(y_true, preds)

        results.append({
            "Fold": fold,
            "Accuracy": acc,
            "AUC": auc,
            "F1-Score": f1
        })

# ✅ 결과 출력
df = pd.DataFrame(results)
df.loc["Average"] = df.mean(numeric_only=True)
print(df.round(4))
