edf_path = "C:\Epilepsy_Project\chb01\chb01_03.edf"

import mne

# EDF 파일 경로
edf_path = "C:\Epilepsy_Project\chb01\chb01_03.edf"

# EDF 로딩
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

# 발작 구간 (summary.txt 참고)
seizure_start = 2996
seizure_end = 3036

# 발작 구간만 자르기
seizure_raw = raw.copy().crop(tmin=seizure_start, tmax=seizure_end)

# 시각화
seizure_raw.plot(duration=10, n_channels=23, title="Seizure EEG")
input("발작 구간 시각화 중... 종료하려면 Enter")
