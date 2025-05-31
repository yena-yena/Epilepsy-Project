import os
import numpy as np
import mne
from tqdm import tqdm
import pywt

patient_list = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]

channel_list = ['Fp1-F7', 'F7-T7', 'T7-P7', 'P7-01', 'Fp2-F8', 'F8-T8', 'T8-P8', 'P8-02']

X_total, y_total = [], []

for patient in patient_list:
    patient_dir = f"./{patient}"
    if not os.path.exists(patient_dir):
        print(f"폴더 없음: {patient_dir}")
        continue

    edf_files = [f for f in os.listdir(patient_dir) if f.endswith('.edf')]
    for fname in tqdm(edf_files, desc = f"처리 중 : {patient}"):
        fpath = os.path.join(patient_dir, fname)
        try:
            raw = mne.io.read_raw_edf(fpath, preload = True, verbose = False)
            raw.pick_channels([ch for ch in channel_list if ch in raw.ch_names])
            data = raw.get_data()

            sfreq = int(raw.info['sfreq'])
            win_sec = 1
            stride_sec = 1
            win_size = win_sec * sfreq
            stride_size = stride_sec * sfreq
            total_len = data.shape[1]

            for start in range(0, total_len - win_size, stride_size):
                segment = data[:, start:start + win_size]

                dwt_result = []
                for ch in segment:
                    coeffs = pywt.wavedec(ch, 'db4', level = 3)
                    dwt_result.append(coeffs[0])

                dwt_result = np.array(dwt_result)
                X_total.append(dwt_result[np.newaxis, ...])

                label = 1 if 'seizure' in fname.lower() else 0
                y_total.append(label)

        except Exception as e:
            print(f"오류 발생 : {fpath} | {str(e)}")

X_total = np.stack(X_total)
y_total = np.array(y_total)

np.save("X_dwt_all.npy", X_total)
np.save("y_total_all.npy", y_total)
print("모든 환자의 데이터를 저장했습니다.")
print("X_dwt_all shape : ", X_total.shape)            
print("y_total_all shape : ", y_total.shape)
