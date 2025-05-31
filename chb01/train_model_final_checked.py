import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 하이퍼파라미터
batch_size = 256
learning_rate = 0.001
epochs = 30

# 2. 데이터 로딩 및 Oversamplings
X = np.load("X_dwt.npy")  # shape: (N, 1, 23, T)
y = np.load("y_total.npy")

X_flat = X.reshape(X.shape[0], -1)
ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y)
X_resampled = X_resampled.reshape(-1, 1, 23, X.shape[3])

# 3. Stratified Split
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# 4. Tensor 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

#######
print("X_train_tensor", X_train_tensor.shape)
print("y_train_tensor", y_train_tensor.shape)
# 5. 모델 정의
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
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)                           # (B, 32, 23, T//4)
        x = x.permute(0, 3, 1, 2)                 # (B, T//4, 32, 23)
        x = x.reshape(x.size(0), x.size(1), -1)   # (B, T//4, 32*23)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x.squeeze()

# 6. 학습 준비
model = EEG_DWT_CNN_LSTM()
pos_weight = torch.tensor([y_train.sum() / (len(y_train) - y_train.sum())])
criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)
train_losses = [] # 손실 저장

# 7. 학습
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

        # === 중간 검증 ===
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_probs = torch.sigmoid(val_outputs)
        val_preds = (val_probs > 0.5).long()
        val_acc = (val_preds == y_val_tensor).sum().item() / len(y_val_tensor) * 100
        print(f"🧪 [Epoch {epoch+1}] Validation Accuracy: {val_acc:.2f}%")

        # Confusion Matrix 출력
        cm = confusion_matrix(y_val_tensor, val_preds)
        print("Confusion Matrix:")
        print(cm)

        # Classification report
        print("Classification Report:")
        print(classification_report(y_val_tensor, val_preds, target_names=["정상", "발작"]))



    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).long()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

        ######
        print("Logits (raw output):", outputs[:5])
        print("Sigmoid outputs:", torch.sigmoid(outputs[:5]))

    train_losses.append(total_loss)
    acc = correct / total * 100
    print(f"[{epoch+1}/{epochs}] Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")

# 8. 검증
plt.plot(train_losses, marker = 'o', color = 'coral')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    probs = torch.sigmoid(val_outputs)  # logits → 확률로 변환


    #### sigmoid 출력 분포 시각화 ####
    plt.hist(probs.numpy(), bins = 50, color = 'skyblue')
    plt.title("Sigmoid Output Distribution (Validation)")
    plt.xlabel("Sigmoid Output")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    ### threshold 조절 실험 ###
    for threshold in [0.5, 0.6, 0.65, 0.7]:    
        val_preds = (probs > threshold).long()
        val_acc = (val_preds == y_val_tensor).sum().item() / len(y_val_tensor) * 100
        print(f"\n Threshold = {threshold}")
        print(f"\n→ Validation Accuracy: {val_acc:.2f}%")
        print("예측값 분포 : ", val_preds.bincount())

        # Confusion Matrix
        cm = confusion_matrix(y_val_tensor, val_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["정상", "발작"])
        disp.plot(cmap=plt.cm.Blues)
        preds = (probs > threshold).long()
        plt.title(f"Confusion Matrix (threshold) = {threshold}")
        plt.show()

        # 상세 지표
        print("Classification Report : ")
        print(classification_report(y_val_tensor, val_preds, target_names = ["정상", "발작"]))
# 9. 저장
torch.save(model.state_dict(), "eeg_final_checked.pt")
print("💾 저장 완료! (eeg_final_checked.pt)")
