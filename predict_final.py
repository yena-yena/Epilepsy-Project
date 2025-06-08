# predict_final.py

import torch
import numpy as np
from train_model_kfold_cnn1d import CNN1DModel
import random

# Fold 1 모델 로드
model = CNN1DModel(input_length=640)  # 8채널 × 80타임포인트
model.load_state_dict(torch.load("backend_server/saved_models/model_fold1_best.pt", map_location="cpu"))
model.eval()

# 데이터 불러오기
X = np.load("backend_server/npy/X_dwt.npy")        # shape: (N, 8, 80)
y = np.load("backend_server/npy/y_total.npy")      # shape: (N,)
ids = np.load("backend_server/npy/id_list.npy")    # shape: (N,)

# Fold 1에 해당하는 환자만 추출
fold1_ids = ["chb01", "chb02", "chb03", "chb04", "chb05"]
idx = np.isin(ids, fold1_ids)

X_fold1 = X[idx]
y_fold1 = y[idx]

# 발작 샘플 중 랜덤 1개 뽑기
seizure_idx = np.where(y_fold1 == 1)[0]
sample_idx = random.choice(seizure_idx)

input_sample = torch.tensor(X_fold1[sample_idx:sample_idx+1], dtype=torch.float32)  # shape: [1, 8, 80]
true_label = y_fold1[sample_idx]

# 예측 수행
with torch.no_grad():
    output = model(input_sample)
    prob = torch.sigmoid(output).item()

print(f"✅ 예측 확률: {prob*100:.2f}% (정답: {true_label})")
