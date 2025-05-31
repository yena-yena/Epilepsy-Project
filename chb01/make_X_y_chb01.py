# ✅ 1. chb01만으로 X_dwt.npy, y_total.npy 만들기
# 파일명: make_X_y_chb01.py

import os
import numpy as np
import mne
import pywt

edf_dir = './chb01'  # chb01만 처리
X_all = []
y_all = []
target_channels = ['Fp1-F7', 'F7-T7']  # 예시 (2채널)

for filename in os.listdir(edf_dir):
    if filename.endswith('.edf'):
        file_path = os.path.join(edf_dir, filename)
        print(f'📂 처리 중: {file_path}')
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.pick_channels([ch for ch in target_channels if ch in raw.ch_names])
        data = raw.get_data()  # shape: (n_channels, n_times)

        for i in range(0, data.shape[1] - 256*5, 256*2):
            window = data[:, i:i+256*5]  # 5초
            coeffs = pywt.wavedec(window, 'db4', level=3, axis=1)
            dwt = np.array(coeffs[1])  # 예시: D3만 사용
            dwt = dwt.reshape(1, len(target_channels), -1)  # (1, 채널, 시간)
            X_all.append(dwt)
            y_all.append(0)  # 임시 라벨: 정상 (나중에 seizure info 파싱 가능)

X_all = np.array(X_all)
y_all = np.array(y_all)

np.save('X_dwt.npy', X_all)
np.save('y_total.npy', y_all)
print('✅ 저장 완료!')
