import torch
import numpy as np
from train_model_kfold_cnn_bilstm import CNNBiLSTMModel

# ✅ 모델 세팅
model = CNNBiLSTMModel(input_channels=8, input_time=80)
model.load_state_dict(torch.load("backend_server/saved_models/model_fold1_best.pt", map_location="cpu"))
model.eval()

# ✅ 데이터 불러오기
X = np.load("backend_server/npy/X_dwt.npy")         # shape: [N, 8, 80]
y = np.load("backend_server/npy/y_total.npy")       # shape: [N]
ids = np.load("backend_server/npy/id_list.npy")     # shape: [N]

# ✅ Fold1 환자 샘플 추출
fold1_ids = ["chb01", "chb02", "chb03", "chb04"]  # Fold1에 해당하는 환자 ID들
idx = np.isin(ids, fold1_ids)
X_fold1 = X[idx]
y_fold1 = y[idx]

# ✅ 발작 샘플 하나 선택 (label == 1)
sample_idx = np.where(y_fold1 == 1)[0][0]
x_sample = X_fold1[sample_idx:sample_idx+1]  # shape: [1, 8, 80]
x_tensor = torch.tensor(x_sample, dtype=torch.float32)

# ✅ 예측 수행
with torch.no_grad():
    output = model(x_tensor).squeeze()
    prob = torch.sigmoid(output).item()

print("\n🎯 예측 결과")
print(f"정답 라벨: {y_fold1[sample_idx]}")
print(f"예측 확률 (발작일 확률): {prob * 100:.2f}%")
