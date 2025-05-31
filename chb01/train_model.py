import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 하이퍼파라미터
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# 데이터 불러오기
X = np.load("X_chb01.npy")
y = np.load("y_chb01.npy")

# 차원 맞추기: (samples, channels, time) → (samples, 1, channels, time)
X = X[:, np.newaxis, :, :]

# 텐서 변환
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
            nn.Flatten(),
            nn.Linear(32 * 23 * (2560 // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

# 모델 초기화
model = EEG_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
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
    print(f"[{epoch+1}/{num_epochs}] Loss: {total_loss:.4f} | Acc: {acc:.2f}%")

print("✅ 모델 학습 완료!")

# 모델 저장
torch.save(model.state_dict(), "eeg_cnn_model.pt")
print("💾 모델 저장 완료! (eeg_cnn_model.pt)")

