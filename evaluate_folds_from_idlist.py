import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

import os

# ===== 한글 폰트 설정 (OS별 자동 적용) =====
if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
else:  # Linux (Colab 등)
    rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 모델 정의
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

# 데이터 로딩
X = np.load("backend_server/npy/X_dwt.npy")
y = np.load("backend_server/npy/y_total.npy")
ids = np.load("backend_server/npy/id_list.npy")

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
        preds = (probs >= 0.5).astype(int)   # threshold=0.5

    # 평가 지표
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = None
    f1 = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = (0, 0, 0, 0)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()

    # ✅ 평가 지표 print
    print(f"\n----- Fold {fold+1} ({', '.join(test_ids)}) -----")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC:       {auc if auc is not None else 'N/A'}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Confusion Matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}")

    # ✅ 혼동행렬 시각화
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Fold {fold+1} 혼동행렬 (임계값=0.5)')
    plt.colorbar()
    classes = ['정상', '발작']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2. if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=16)

    plt.ylabel('실제값', fontsize=12)
    plt.xlabel('예측값', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_fold{fold+1}.png")  # 이미지 파일로 저장
    plt.show()

    # csv 결과 저장용
    results.append({
        "Fold": fold + 1,
        "Test Patients": ", ".join(test_ids),
        "Accuracy": acc,
        "AUC": auc,
        "F1-Score": f1,
        "Precision": precision,
        "Recall": recall,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

# 평균 추가
df = pd.DataFrame(results)
mean_row = df[["Accuracy", "AUC", "F1-Score", "Precision", "Recall", "TN", "FP", "FN", "TP"]].mean()
mean_row["Fold"] = "평균"
mean_row["Test Patients"] = ""
df = df.append(mean_row, ignore_index=True)
df = df.round(4)
print(df)
df.to_csv("evaluation_fold_results.csv", index=False)
print("\n✅ 모든 fold 지표와 혼동행렬 이미지 저장 완료!")
