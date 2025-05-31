import mne

# 예시 파일 경로 (받은 파일 경로로 수정!)
edf_path = "C:/Epilepsy_Project/chb01_01.edf"

# EDF 파일 로드
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False) 

# 채널 이름 확인
print("채널 목록:", raw.ch_names)

# 시각화 (20초)
raw.plot(duration=20, n_channels=23)
input("시각화 창이 열려있습니다. 종료하려면 Enter를 누르세요.")

### chb01_01.edf 불러오기 성공, 채널 목록 확인됨, EEG 시각화 완료, input()으로 창 유지 완료료