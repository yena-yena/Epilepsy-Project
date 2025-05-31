import os
import numpy as np

data_dir = "C:\Epilepsy_Project\chb01"

X_list = []
y_list = []

for fname in os.listdir(data_dir):
    path = os.path.join(data_dir, fname)
    
    if fname.startswith("X_") and fname.endswith(".npy"):
        x = np.load(path)
        
        # ✅ 차원 안 맞으면 맞춰줌 (3D → 4D로 reshape)
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]  # (N, C, T) → (N, 1, C, T)
        X_list.append(x)

    elif fname.startswith("y_") and fname.endswith(".npy"):
        y = np.load(path)
        y_list.append(y)

X_total = np.concatenate(X_list, axis=0)
y_total = np.concatenate(y_list, axis=0)


np.save(os.path.join(data_dir, "X_total.npy"), X_total)
np.save(os.path.join(data_dir, "y_total.npy"), y_total)


print(f"✅ 합치기 완료! X shape: {X_total.shape}, y shape: {y_total.shape}")
print(f"🧠 발작: {np.sum(y_total)}, 정상: {len(y_total)-np.sum(y_total)}")
