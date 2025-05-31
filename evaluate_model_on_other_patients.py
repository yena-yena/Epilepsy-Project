import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

base_path = "C:/Epilepsy_Project"
target_folders = [f for f in os.listdir(base_path) if f.startswith("chb")]

for folder in target_folders:
    x_old = os.path.join(base_path, folder, "X_dwt.npy")
    y_old = os.path.join(base_path, folder, "y_total.npy")
    x_new = os.path.join(base_path, folder, "X.npy")
    y_new = os.path.join(base_path, folder, "y.npy")
    
    if os.path.exists(x_old):
        os.rename(x_old, x_new)
    if os.path.exists(y_old):
        os.rename(y_old, y_new)

print("이름 변경 완료!")


# 모델 구조 동일하게 정의
class EEG_DWT_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((1, 2))
        )
        self.lstm = nn.LSTM(input_size=32 * 23, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.cnn(x)                           # (B, 32, 23, T//4)
        x = x.permute(0, 3, 1, 2)                 # (B, T//4, 32, 23)
        x = x.reshape(x.size(0), x.size(1), -1)   # (B, T//4, 32*23)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x.squeeze()


# 모델 불러오기
model = EEG_DWT_CNN_LSTM()
model.load_state_dict(torch.load("eeg_final_checked.pt"))
model.eval()

# 평가할 환자 리스트
patient_list = ["chb02", "chb03", "chb05", "chb08", "chb09"]  # 필요하면 더 추가!

# 평가 루프
for patient_id in patient_list:
    x_path = f"./{patient_id}/X.npy"
    y_path = f"./{patient_id}/y.npy"

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"{patient_id} 데이터 없음. 스킵.")
        continue

    # 데이터 로딩
    X = np.load(x_path)
    y = np.load(y_path)

    # 텐서 변환
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)

        print(f"\n=== Evaluation on {patient_id} ===")
        for threshold in [0.5, 0.6, 0.65, 0.7]:
            preds = (probs > threshold).long()
            acc = (preds == y_tensor).sum().item() / len(y_tensor) * 100

            print(f"\n[Threshold: {threshold}] Accuracy: {acc:.2f}%")
            print("Confusion Matrix:")
            print(confusion_matrix(y_tensor, preds))
            print("Report:")
            print(classification_report(y_tensor, preds, target_names=["정상", "발작"]))

            # 시각화
            cm = confusion_matrix(y_tensor, preds)
            disp = ConfusionMatrixDisplay(cm, display_labels=["정상", "발작"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"{patient_id} - Confusion Matrix (th={threshold})")
            plt.show()
