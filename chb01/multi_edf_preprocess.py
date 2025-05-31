# EDF 여러 개 처리

import mne
import numpy as np
import os

# 💾 EDF 파일 리스트 (파일명, 발작 시작시간, 발작 끝시간)
edf_list = [
    ("C:\Epilepsy_Project\chb01\chb01_03.edf", 2996, 3036),
    ("C:\Epilepsy_Project\chb01\chb01_04.edf", 1467, 1494),
    ("C:\Epilepsy_Project\chb01\chb01_15.edf", 1732, 1772),
    ("C:\Epilepsy_Project\chb01\chb01_16.edf", 1015, 1066),
    ("C:\Epilepsy_Project\chb01\chb01_18.edf", 1720, 1810),
]

# 슬라이딩 파라미터
sampling_rate = 256
window_sec = 10
step_sec = 5
window_size = window_sec * sampling_rate
step_size = step_sec * sampling_rate

# 결과 저장 리스트
X_all = []
y_all = []

for filename, sz_start, sz_end in edf_list:
    print(f"📁 Processing {filename}...")
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
    data, times = raw.get_data(return_times=True)  # data: (channels, samples)

    total_samples = data.shape[1]

    for start in range(0, total_samples - window_size + 1, step_size):
        end = start + window_size
        segment = data[:, start:end]
        t_start = times[start]
        t_end = times[end - 1]

        # 🧠 발작 구간 겹치면 라벨 1, 아니면 0
        if sz_start <= t_end and sz_end >= t_start:
            label = 1
        else:
            label = 0

        X_all.append(segment)
        y_all.append(label)

# 최종 데이터 배열로 변환
X = np.array(X_all)[:, np.newaxis, :, :]  # (N, 1, 채널, 길이)
y = np.array(y_all)

# 저장
np.save("X_total.npy", X)
np.save("y_total.npy", y)

print("✅ 전체 처리 완료!")
print(f"📐 X shape: {X.shape}, y shape: {y.shape}")
print(f"🧠 발작 라벨 수: {np.sum(y)}, 정상 라벨 수: {len(y) - np.sum(y)}")
