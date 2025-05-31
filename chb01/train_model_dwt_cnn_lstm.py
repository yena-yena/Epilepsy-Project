import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 하이퍼파라미터
batch_size = 500
learning_rate = 0.001
epochs = 10

# 1. 데이터 로딩 및 오버샘플링
X = np.load("X_dwt.npy")  # (N, 1, 23, T)
y = np.load("y_total.npy")  # (N,)

X_flat = X.reshape(X.shape[0], -1)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y)
X_resampled = X_resampled.reshape(-1, 1, 23, X.shape[3])

# 2. Stratify 기반 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# 3. Tensor 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

# 4. 모델 정의
class EEG_DWT_CNN_LSTM(nn.Module):
    def __init__(self):
        super(EEG_DWT_CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),      # zero padding
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),     # zero padding
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
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)                          # (B, 32, 23, T//4)
        x = x.permute(0, 3, 1, 2)                # (B, T//4, 32, 23)
        x = x.reshape(x.size(0), x.size(1), -1)  # (B, T//4, 32*23)
        x, _ = self.lstm(x)
        x = x[:, -1, :]                          # 마지막 timestep
        x = self.fc(x)
        return x.squeeze()

# 5. 학습 준비
model = EEG_DWT_CNN_LSTM()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).long()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total * 100
    print(f"[{epoch+1}/{epochs}] Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")

# 7. 검증
model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_preds = (val_outputs > 0.5).long()
    val_acc = (val_preds == y_val_tensor).sum().item() / len(y_val_tensor) * 100
    print(f"\n→ Validation Accuracy: {val_acc:.2f}%")

    cm = confusion_matrix(y_val_tensor, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["정상", "발작"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (검증 데이터 기준)")
    plt.savefig("confusion_matrix_dwt_oversampled.png")
    plt.show()

# 8. 저장
torch.save(model.state_dict(), "eeg_dwt_cnn_lstm_oversampled.pt")
print("💾 저장 완료! (eeg_dwt_cnn_lstm_oversampled.pt)")
