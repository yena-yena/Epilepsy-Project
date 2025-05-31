import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용중 : ", device)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.ioff()  # 비대화형 모드로 전환

# 1. 하이퍼파라미터
batch_size = 32
learning_rate = 5e-4  # 개선: 3e-4에서 5e-4로 조정
epochs = 50

# 2. 데이터 로딩 및 Oversampling
X = np.load("npy/X_dwt.npy")  # shape: (N, 1, C, T)
y = np.load("npy/y_total.npy")
print("클래스 분포 :", np.unique(y, return_counts=True))

# 개선: 채널별 정규화 (Frontiers, 2020 참고)
X = (X - np.mean(X, axis=(0, 2, 3), keepdims=True)) / (np.std(X, axis=(0, 2, 3), keepdims=True) + 1e-6)

X_flat = X.reshape(X.shape[0], -1)
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y)

y_resampled = y_resampled.astype(np.int64)
print("X_resampled shape : ", X_resampled.shape)
print("y_resampled unique : ", np.unique(y_resampled, return_counts=True))
_, _, C, T = X.shape
X_resampled = X_resampled.reshape(-1, 1, C, T)

# 3. Stratified Split
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)
print("Train y :", np.unique(y_train, return_counts=True))
print("Val y :", np.unique(y_val, return_counts=True))

# 4. Tensor 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

print("검증 배치 개수 : ", len(val_loader))

# 5. 모델 정의
class EEG_DWT_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),  # 개선: 0.1에서 0.25로 조정
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.lstm = nn.LSTM(input_size=272, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25),  # 개선: 0.2에서 0.25로 조정
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)        
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# 6. 학습 준비
model = EEG_DWT_CNN_LSTM().to(device)

# 파라미터 초기화
for param in model.parameters():
    if param.requires_grad and param.dim() > 1:
        nn.init.kaiming_normal_(param)
print("초기화 끗 자 예나야 이제부터 기도 메타다 빌어라 빌어")

# pos_weight 계산 및 조정
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
pos_weight = torch.tensor([n_neg / (n_pos + 1e-6) * 0.9], dtype=torch.float32).to(device)  # 개선: *0.9 추가
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
train_accuracies = []

# 학습 루프 시작 전 첫 배치 sigmoid 출력
first_batch = next(iter(train_loader))
debug_x, debug_y = first_batch
debug_x = debug_x.to(device)
with torch.no_grad():
    debug_outputs = model(debug_x)
    print("CNN Output : ", model.cnn(debug_x).shape)
    print("첫번째 배치 output (logits) : ", debug_outputs[:10].view(-1))
    print("sigmoid result : ", torch.sigmoid(debug_outputs[:10].view(-1)))
    
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0

    for batch_x, batch_y in train_loader:
        batch_count += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float()
        optimizer.zero_grad()
        outputs = model(batch_x)
        if batch_count == 1:  # 첫 배치에서만 CNN 출력 크기 표시
            print("CNN Output : ", model.cnn(batch_x).shape)
        outputs = outputs.view(-1).float()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        total += batch_x.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == batch_y.long()).sum().item()

    avg_loss = total_loss / total
    acc = correct / total * 100
    train_losses.append(avg_loss)
    train_accuracies.append(acc)
    print(f"[{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Train Accuracy: {acc:.2f}%")

    if (epoch + 1) % 10 == 0:  # 10, 20, 30, 40, 50번째 에포크에서 검증
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            print("CNN Output : ", model.cnn(X_val_tensor).shape)
            val_probs_hist = torch.sigmoid(val_logits).cpu().numpy()  # 히스토그램용
            
            plt.figure(figsize=(8, 5))
            plt.hist(val_probs_hist, bins=50, color='skyblue')
            plt.title(f"Epoch {epoch + 1} - Sigmoid Output Histogram")
            plt.xlabel("Sigmoid Output")
            plt.ylabel("Count")
            plt.grid(True)
            plt.show()
            plt.close()
            
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                outputs = model(val_x)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                val_preds.append(preds.cpu().view(-1))
                val_targets.append(val_y.cpu().view(-1))
                val_probs.append(probs.cpu().view(-1))
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_probs = torch.cat(val_probs)
        val_acc = (val_preds == val_targets).float().mean().item() * 100

        print(f"Validation accuracy : {val_acc:.2f}%")
        
        # Confusion Matrix
        cm = confusion_matrix(val_targets.numpy(), val_preds.numpy())
        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
        plt.title(f"Epoch {epoch + 1} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
        plt.close()
        
        # Classification Report
        print("Classification Report:")
        print(classification_report(val_targets.numpy(), val_preds.numpy(), target_names=["Class 0", "Class 1"]))
        
        # 개선: ROC-AUC 추가
        roc_auc = roc_auc_score(val_targets.numpy(), val_probs.numpy())
        print(f"ROC-AUC: {roc_auc:.4f}")

# 8. 모델 저장
torch.save(model.state_dict(), "eeg_model_smote_v2.pt")
print("💾 모델 저장 완료: eeg_model_smote_v2.pt")

# 9. 로그 저장
with open("train_log_improved.txt", "w") as f:
    for i in range(len(train_losses)):
        f.write(f"[{i + 1}/{epochs}] Loss : {train_losses[i]:.4f} | {train_accuracies[i]:.2f}%\n")
    f.write(f"Validation accuracy : {val_acc:.2f}%\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")