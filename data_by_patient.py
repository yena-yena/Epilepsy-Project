import numpy as np
from collections import defaultdict

# 파일 로드
X = np.load("npy/X_dwt.npy")           # shape: (N, 1, ch, T)
y = np.load("npy/y_total.npy")         # shape: (N,)
ids = np.load("npy/id_list.npy")       # shape: (N,)

# 환자 ID별 데이터 묶기
data_by_patient = defaultdict(lambda: [[], []])  # { 'chb01': [[X], [y]], ... }

for i in range(len(X)):
    pid = ids[i]
    data_by_patient[pid][0].append(X[i])
    data_by_patient[pid][1].append(y[i])

# 리스트를 numpy array로 변환
for pid in data_by_patient:
    data_by_patient[pid] = (
        np.stack(data_by_patient[pid][0]),  # X
        np.array(data_by_patient[pid][1])   # y
    )

print("환자별 샘플 수:")
for pid, (X_, y_) in data_by_patient.items():
    print(f"{pid}: {len(X_)}개")
