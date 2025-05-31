import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용중 : ", device)
#

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 하이퍼파라미터
batch_size = 32
learning_rate = 1e-5
epochs = 50

# 2. 데이터 로딩 및 Oversampling
X = np.load("npy/X_dwt.npy")  # shape: (N, 1, C, T)
y = np.load("npy/y_total.npy")
print("클래스 분포 :", np.unique(y, return_counts = True))


X_flat = X.reshape(X.shape[0], -1)
smote = SMOTE(sampling_strategy = 1.0, random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y)

y_resampled = y_resampled.astype(np.int64)
print("X_resampled shape : ", X_resampled.shape)
print("y_resampled unique : ", np.unique(y_resampled, return_counts = True))
_, _, C, T = X.shape
X_resampled = X_resampled.reshape(-1, 1, C, T)
# ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)
# X_resampled, y_resampled = ros.fit_resample(X_flat, y)
# X_resampled = X_resampled.reshape(-1, 1, X.shape[2], X.shape[3])
# _, _, C, T = X.shape
# X_resampled = X_resampled.reshape(-1, 1, C, T)

# print("Oversampled class distribution : ", np.unique(y_resampled, return_counts = True))
# X_flat_check = X_resampled.reshape(X_resampled.shape[0], -1)
# X_flat_unique = np.unique(X_flat_check, axis = 0)
# print(f"전체 샘플 수 : {len(X_flat_check)}")
# print(f"유일한 샘플 수 : {len(X_flat_unique)}")
# print(f"중복 비율 : {100 * (1 - len(X_flat_unique) / len(X_flat_check)):.2f}%")
# 3. Stratified Split
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)
#
print("Train y :", np.unique(y_train, return_counts = True))
print("Val y :", np.unique(y_val, return_counts = True))

# 4. Tensor 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

#
print("검증 배치 개수 : ", len(val_loader))
# 5. 모델 정의
class EEG_DWT_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d((1, 2)),

            # 추가 #
            nn.Conv2d(32, 16, kernel_size = 1)
        )
        # self.lstm = nn.LSTM(input_size=32 * X.shape[2], hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size = 272, hidden_size = 64, batch_first = True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 17, 128),
            #nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            #nn.Dropout(0.1),
            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        #
        # print("CNN Output : ", x.shape)
        #
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = x.permute(0, 2, 1, 3)
        #x = x.reshape(x.size(0), x.size(1), -1)
        #
        # print("LSTM Input Shape : ", x.shape)
        #
        #x, _ = self.lstm(x)
        #x = self.fc(x[:, -1, :])
        return x


# 6. 학습 준비
model = EEG_DWT_CNN_LSTM().to(device)

# 추가 ; 파라미터 초기화
for param in model.parameters():
   if param.requires_grad and param.dim()> 1:
        nn.init.kaiming_normal_(param)
print("초기화 끗 자 예나야 이제부터 기도 메타다 빌어라 빌어")
# pos_weight 계산
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
# pos_weight_value = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)
# pos_weight = torch.tensor([1.5], dtype = torch.float32).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
train_accuracies = []
# 7. 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        print("배치 y 분포 : ", batch_y[:10])
        break

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.float()
        outputs = model(batch_x)
    
        if epoch == 0:
            print("첫 번째 배치 output (logits) : ", outputs[:10].view(-1))
            print("sigmoid 결과 : ", torch.sigmoid(outputs[:10].view(-1)))
            break

        # 정상인걸로 확인 됨
        #print(outputs.shape, outputs.shape)
        #print("batch_y.shape", batch_y.shape)
        #print("Sigmoid outputs (첫 5개): ", torch.sigmoid(outputs[:5]))
        #print("Targets (이하동일) :", batch_y[:5])
        #

        outputs = outputs.view(-1).float()
        batch_y = batch_y.view(-1).float()
        loss = criterion(outputs, batch_y)
        outputs = torch.sigmoid(model(batch_x)).view(-1)
        #loss_fn = torch.nn.BCELoss()
        #loss = loss_fn(outputs, batch_y.view(-1))


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == batch_y.long()).sum().item()
        total += batch_y.size(0)

        if total > 0:
            acc = correct / total * 100
        else:
            acc = 0.0
        print(f"[{epoch+1}/{epochs}] Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")
        train_losses.append(total_loss)
        train_accuracies.append(acc)
        break


    if (epoch + 1) % 10 == 0 or (epoch + 1 ) == epochs:
        val_preds = []
        val_targets = []

        with torch.no_grad():
            model.eval()
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                outputs = model(val_x)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()

                val_preds.append(preds.cpu().view(-1))
                val_targets.append(val_y.cpu().view(-1))

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_correct = (val_preds == val_targets).sum().item()
        val_total = val_targets.size(0)
        val_acc = val_correct / val_total * 100

        print(f"Validation accuracy : {val_acc:.2f}%")
            ##추가##
            #val_logits = model(X_val_tensor.to(device))
            #
            #print(val_logits[:10])
            #val_probs = torch.sigmoid(val_logits)

            ######
            #print("Sigmoid m :", val_probs.min().item())
            #print("Sigmoid M :", val_probs.max().item())
            #print("Sigmoid E :", val_probs.mean().item())

            #val_preds = (val_probs > 0.5).long()
            #val_labels = y_val_tensor.to(device)

            #val_correct = (val_preds == y_val_tensor).sum().item()
            #val_total = y_val_tensor.to(device).size(0)
            #val_acc = val_correct / val_total * 100
            #print(f"Validation accuracy : {val_acc * 100:.2f}%")

            ##추가##
            #plt.hist(val_probs.cpu().numpy(), bins = 50, color = 'skyblue')
            #plt.title(f"Epoch {epoch + 1} Sigmoid 분포")
            #plt.xlabel("Sigmoid Output")
            #plt.ylabel("Count")
            #plt.grid(True)
            #plt.show()

# 8. 모델 저장
torch.save(model.state_dict(), "eeg_model_smote_v1.pt")
print("💾 모델 저장 완료: eeg_model_smote_v1.pt")

# 9. 로그 저장
with open("train_log.txt", "w") as f:
    for i in range(len(train_losses)):
        f.write(f"[{i + 1} / {epochs}] Loss : {train_losses[i]:.4f} | {train_accuracies[i]:.2f}%|n")
    f.write(f"Validation accuracy : {val_acc:.2f}%n")