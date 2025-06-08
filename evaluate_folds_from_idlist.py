import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import os

# ✅ 예나님 학습 모델 구조 그대로 복붙
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
            x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x

# ✅ 데이터 로딩
X = np.load("backend_server/npy/X_dwt.npy")
y = np.load("backend_server/npy/y_total.npy")
ids = np.load("backend_server/npy/id_list.npy")

# ✅ Fold 나누기
unique_ids = sorted(list(set(ids)))
n_folds = 4
fold_size = len(unique_ids) // n_folds
results = []

for fold in range(n_folds):
    model_path = f"backend_server/saved_models/model_fold{fold+1}_best.pt"
    if not os.path.exists(model_path):
        print(f"❗ model_fold{fold+1}_best.pt 없음 → 건너뜀")
        continue

    test_ids = unique_ids[fold * fold_size : (fold + 1) * fold_size]
    test_idx = np.isin(ids, test_ids)
    X_test = X[test_idx]
    y_test = y[test_idx]

    model = CNNBiLSTMModel(input_channels=8, input_time=80)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs).numpy()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)

    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        print(f"⚠️ Fold {fold+1}: AUC 계산 실패 (한쪽 클래스만 존재)")
        auc = None

    f1 = f1_score(y_test, preds)

    results.append({
        "Fold": fold + 1,
        "Test Patients": ", ".join(test_ids),
        "Accuracy": acc,
        "AUC": auc,
        "F1-Score": f1
    })

# ✅ 결과 정리
df = pd.DataFrame(results)
df.loc["Average"] = df[["Accuracy", "AUC", "F1-Score"]].dropna().mean()
print(df.round(4))
