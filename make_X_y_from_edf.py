import os
import mne
import numpy as np
import pywt

# 설정
data_dir = r"C:\\Users\\user\\Desktop\\프로젝트\\Epilepsy-Project"
output_dir = os.path.join(data_dir, "npy")
os.makedirs(output_dir, exist_ok=True)

X_list = []
y_list = []
id_list = []

window_sec = 2
stride_sec = 1
sfreq = 256
window = window_sec * sfreq
stride = stride_sec * sfreq

# 사용할 채널
target_channels = ['FP1-F7', 'FP2-F8', 'F7-T7', 'T7-P7', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1']

# 수동 라벨링된 발작 파일 목록
seizure_files = {
    "chb01": ["chb01_03.edf", "chb01_04.edf", "chb01_16.edf", "chb01_18.edf", "chb01_21.edf"],
    "chb02": ["chb02_16.edf"],
    "chb03": ["chb03_01.edf", "chb03_02.edf", "chb03_03.edf"],
    "chb05": ["chb05_06.edf"],
    "chb08": ["chb08_02.edf"],
    "chb09": ["chb09_06.edf"]
}

# 환자 폴더 순회
for patient_folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, patient_folder)
    if not os.path.isdir(folder_path) or not patient_folder.startswith("chb"):
        continue

    for fname in os.listdir(folder_path):
        if not fname.endswith(".edf"):
            continue

        file_path = os.path.join(folder_path, fname)
        print(f"\n📂 처리 중: {file_path}")

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            picked_channels = [ch for ch in target_channels if ch in raw.ch_names]
            raw.pick_channels(picked_channels)
            data = raw.get_data()  # (채널 수, 시간)

            # 채널 수 부족 시 zero-padding
            if len(picked_channels) < 8:
                pad_count = 8 - len(picked_channels)
                padding = np.zeros((pad_count, data.shape[1]))
                data = np.vstack((data, padding))
            elif len(picked_channels) > 8:
                data = data[:8]  # 너무 많으면 자르기

            label = 1 if fname in seizure_files.get(patient_folder, []) else 0

        except Exception as e:
            print(f"⚠️ 오류로 건너뜀: {fname} ({e})")
            continue

        for start in range(0, data.shape[1] - window + 1, stride):
            seg = data[:, start:start+window]
            dwt_seg = []
            for ch in seg:
                coeffs = pywt.wavedec(ch, 'db4', level=3)
                cA3 = coeffs[0]  # 저주파 계수
                # 길이를 38로 맞추기 (부족하면 패딩, 넘치면 자르기)
                if len(cA3) < 38:
                    padded = np.pad(cA3, (0, 38 - len(cA3)), mode='constant')
                else:
                    padded = cA3[:38]
                dwt_seg.append(padded)
            dwt_seg = np.array(dwt_seg)

            if dwt_seg.shape != (8, 38):
                print(f"⚠️ 스킵됨: {dwt_seg.shape}")
                continue

            X_list.append(dwt_seg[np.newaxis, :, :])  # (1, 8, 38)
            y_list.append(label)
            id_list.append(patient_folder)

# 최종 저장
X = np.stack(X_list)
y = np.array(y_list)
ids = np.array(id_list)

np.save(os.path.join(output_dir, "X_dwt.npy"), X)
np.save(os.path.join(output_dir, "y_total.npy"), y)
np.save(os.path.join(output_dir, "id_list.npy"), ids)

print("\n✅ 저장 완료!")
print("📐 X_dwt shape:", X.shape)
print("📐 y_total shape:", y.shape)
print("📐 id_list shape:", ids.shape)