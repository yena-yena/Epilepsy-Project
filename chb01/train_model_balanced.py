import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# 🧸 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 기준
plt.rcParams["axes.unicode_minus"] = False

# 하이퍼파라미터
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# 1. 데이터 불러오기
X = np.load("X_total.npy")
y = np.load("y_total.npy")
print(f"전체 데이터 수: {len(X)}, 발작 수: {np.sum(y)}")

# 2. train/validation 나누기 (30% 검증)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 3. train만 oversample
X_train_flat = X_train.reshape(len(X_train), -1)
ros = RandomOverSampler(sampling_strategy=0.6, random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_flat, y_train)
X_train = X_train_resampled.reshape(-1, 1, 23, 2560)

# 4. 텐서 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 5. DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

# 6. 모델 정의
class EEG_CNN_LSTM(nn.Module):
    def __init__(self):
        super(EEG_CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.lstm = nn.LSTM(input_size=32 * 23, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)  # (B, T, C, H)
        x = x.reshape(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x

model = EEG_CNN_LSTM()

# 7. 가중치 설정
n_pos = sum(y_train_resampled)
n_neg = len(y_train_resampled) - n_pos
w_neg = 1
w_pos = (n_neg / n_pos) * 1.5
weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 8. 학습
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total * 100
    print(f"[{epoch+1}/{num_epochs}] Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")

# 9. 검증
model.eval()
with torch.no_grad():
    val_preds = model(X_val_tensor).argmax(dim=1)

val_acc = (val_preds == y_val_tensor).sum().item() / len(y_val_tensor) * 100
print(f"\n✅ 검증 정확도: {val_acc:.2f}%")

# 10. 혼동 행렬
cm = confusion_matrix(y_val_tensor, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["정상", "발작"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (검증 데이터 기준)")
plt.savefig("confusion_matrix_val_korean.png")
plt.show()

# 11. 정밀지표 출력
print("\n📊 Classification Report:")
print(classification_report(y_val_tensor, val_preds, target_names=["정상", "발작"]))

# 12. 저장
torch.save(model.state_dict(), "eeg_cnn_oversampled_korean.pt")
print("💾 저장 완료! (eeg_cnn_oversampled_korean.pt)")
