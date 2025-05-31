import mne
import numpy as np

# 파라미터 설정
edf_path = "chb01_03.edf"
seizure_start = 2996
seizure_end = 3036
window_size_sec = 10
step_size_sec = 5  # 겹치게 하고 싶지 않으면 10으로 설정
sampling_rate = 256  # Hz

# EDF 불러오기
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
data, times = raw.get_data(return_times=True)  # shape: (channels, samples)

# 슬라이딩 윈도우 준비
window_size = window_size_sec * sampling_rate
step_size = step_size_sec * sampling_rate
total_samples = data.shape[1]

X = []
y = []

# 윈도우 단위로 자르기
for start in range(0, total_samples - window_size + 1, step_size):
    end = start + window_size
    window_data = data[:, start:end]  # (channels, window_size)

    # 라벨링: 이 윈도우가 발작 시간을 포함하는가?
    start_time = times[start]
    end_time = times[end - 1]

    if seizure_start <= end_time and seizure_end >= start_time:
        label = 1  # 발작 포함
    else:
        label = 0  # 정상

    X.append(window_data)
    y.append(label)

X = np.array(X)  # shape: (num_windows, channels, window_size)
y = np.array(y)  # shape: (num_windows,)

print(f"✅ 윈도우 수: {len(X)}")
print(f"📐 X shape: {X.shape}, y shape: {y.shape}")
print(f"🧠 발작 라벨 수 (1): {np.sum(y)}, 정상 (0): {len(y) - np.sum(y)}")


# 데이터 저장
np.save("X_chb01.npy", X)
np.save("y_chb01.npy", y)

print("✅ X, y 데이터 저장 완료 (X_chb01.npy / y_chb01.npy)")
