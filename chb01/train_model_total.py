import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 하이퍼파라미터
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# 데이터 로딩
X = np.load("X_total.npy")  # (N, 1, 23, 2560)
y = np.load("y_total.npy")  # (N,)


# 텐서로 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# DataLoader 만들기
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# CNN 모델 정의
class EEG_CNN(nn.Module):
    def __init__(self):
        super(EEG_CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 23 * (2560 // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
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

        # CNN → LSTM으로 보낼 때 input size 계산 (32 feature map * 23채널)
        self.lstm = nn.LSTM(input_size=32 * 23, hidden_size=64, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),  # Bidirectional LSTM이니까 64 * 2
            nn.ReLU(),
            nn.Linear(64, 2)  # 클래스 수에 맞게 수정 (2개면 이대로)
        )

    def forward(self, x):
        x = self.cnn(x) # (8, 32, 23, 640)
        x = x.permute(0, 3, 1, 2) # (8, 640, 32, 23)
        x = x.reshape(x.size(0), x.size(1), -1) # (8, 640, 736)

        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]

        x = self.fc(x)
        return x

# 모델 정의
model = EEG_CNN_LSTM()
# 클래스 비율 기반 가중치 계산
total = len(y)
n_pos = np.sum(y)
n_neg = total - n_pos

w_neg = total / (2 * n_neg)
w_pos = total / (2 * n_pos)

weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)

# 수정된 손실 함수 적용
criterion = nn.CrossEntropyLoss(weight=weights)

# 옵티마이저는 그대로
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
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
    print(f"[{epoch+1}/{num_epochs}] Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# 모델 저장
torch.save(model.state_dict(), "eeg_cnn_total.pt")
print("💾 저장 완료! (eeg_cnn_total.pt)")

# 시각화 하기 위함

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 전체 데이터로 다시 예측 (학습용이라도 예시로)
model.eval()
with torch.no_grad():
    preds = model(X_tensor).argmax(dim=1)

cm = confusion_matrix(y_tensor, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["정상", "발작"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (학습 데이터 기준)")
plt.savefig("confusion_matrix.png")
plt.show()
