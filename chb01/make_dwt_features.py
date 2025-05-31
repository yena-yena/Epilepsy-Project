import numpy as np
import pywt

# 원본 EEG 불러오기
X = np.load("X_total.npy")  # shape: (N, 1, 23, 2560)

def apply_dwt_to_eeg(eeg_sample, wavelet='db4', level=3):
    _, channels, timesteps = eeg_sample.shape
    output = []

    for ch in range(channels):
        signal = eeg_sample[0, ch, :]
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        approx = coeffs[0]  # approximation coefficients
        output.append(approx)

    output = np.stack(output)  # shape: (23, T)
    return output

# 전체 EEG에 DWT 적용
X_dwt_list = []

for i in range(X.shape[0]):
    dwt_feat = apply_dwt_to_eeg(X[i])  # (23, T)
    X_dwt_list.append(dwt_feat)

X_dwt = np.stack(X_dwt_list)  # (N, 23, T)
X_dwt = np.expand_dims(X_dwt, axis=1)  # (N, 1, 23, T)

np.save("X_dwt.npy", X_dwt)
print("✅ DWT 적용 완료! 저장된 파일: X_dwt.npy | Shape:", X_dwt.shape)
