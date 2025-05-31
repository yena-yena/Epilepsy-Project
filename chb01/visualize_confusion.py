import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # or AppleGothic on Mac
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
# 기존 코드
X = np.load("preproccessed_npy/X_total.npy")
y = np.load("preproccessed_npy/y_total.npy")

# 여기!! 추가
print("총 데이터 수:", len(y))
print("발작 라벨 수 (1):", np.sum(y))
print("정상 라벨 수 (0):", len(y) - np.sum(y))

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# CNN 모델 정의 (동일)
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

# 모델 불러오기
model = EEG_CNN()
model.load_state_dict(torch.load("eeg_cnn_total.pt"))
model.eval()

# 예측 + confusion matrix
with torch.no_grad():
    preds = model(X_tensor).argmax(dim=1)

cm = confusion_matrix(y_tensor, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["정상", "발작"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (학습 데이터 기준)")
plt.savefig("confusion_matrix.png")
plt.show()
