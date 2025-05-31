import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -------------------------
# 하이퍼파라미터
# -------------------------
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# -------------------------
# 데이터 로딩 및 오버샘플링
# -------------------------
X = np.load("X_total.npy")  # (N, 1, 23, 2560)
y = np.load("y_total.npy")  # (N,)

# 1. Train/Val 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Train만 오버샘플링 (발작 클래스 비율 20%로)
X_train_flat = X_train.reshape(len(X_train), -1)
ros = RandomOverSampler(sampling_strategy=0.6, random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_flat, y_train)
X_train = X_train_resampled.reshape(-1, 1, 23, 2560)

# 3. Tensor 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 4. 클래스 비율 기반 가중치
n_pos = sum(y_train_resampled)
n_neg = len(y_train_resampled) - n_pos
w_neg = 1
w_pos = (n_neg / n_pos) * 1.5 # 가중치 세게 줌
# 4. DataLoader 구성

weights = torch.tensor([w_neg, w_pos], dtype = torch.float32)
criterion = nn.CrossEntropyLoss(weight = weights)
# -------------------------
# 모델 정의
# -------------------------
class EEG_CNN_LSTM(nn.Module):
    def __init__(self):
        super(EEG_CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(16, 32, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.lstm = nn.LSTM(input_size=32 * 23, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.cnn(x)                          # (B, 32, 23, 640)
        x = x.permute(0, 3, 1, 2)                # (B, 640, 32, 23)
        x = x.reshape(x.size(0), x.size(1), -1)  # (B, 640, 736)
        lstm_out, _ = self.lstm(x)               # (B, 640, 128)
        x = lstm_out[:, -1, :]                   # (B, 128)
        x = self.fc(x)                           # (B, 2)
        return x

model = EEG_CNN_LSTM()

# 클래스 비율 기반 가중치
n_pos = np.sum(y_train_resampled)
n_neg = len(y_train_resampled) - n_pos
w_pos = len(y_train_resampled) / (2 * n_pos)
w_neg = len(y_train_resampled) / (2 * n_neg)
weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)

# 손실함수 & 옵티마이저
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# 학습 루프
# -------------------------
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    acc = correct / total * 100
    print(f"[{epoch+1}/{num_epochs}] Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")

# -------------------------
# 검증 정확도 확인
# -------------------------
model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_preds = val_outputs.argmax(dim=1)
    val_acc = (val_preds == y_val_tensor).sum().item() / len(y_val_tensor) * 100
    print(f"\n→ Validation Accuracy: {val_acc:.2f}%\n")

# -------------------------
# 혼동행렬 출력
# -------------------------
cm = confusion_matrix(y_val_tensor, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["정상", "발작"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (검증 데이터 기준)")
plt.savefig("confusion_matrix_val.png")
plt.show()

# -------------------------
# 모델 저장
# -------------------------
torch.save(model.state_dict(), "eeg_cnn_oversampled.pt")
print("💾 저장 완료! (eeg_cnn_oversampled.pt)")
