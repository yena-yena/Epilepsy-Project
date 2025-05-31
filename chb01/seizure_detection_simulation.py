import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# 시간 정의 (실시간 데이터를 위한 2초 길이)
time = np.linspace(0, 2, 2560)  # 2초 동안 2560개의 샘플

# 예시 EEG 신호 (진동 + 잡음)
eeg_signal = np.sin(2 * np.pi * 10 * time) + 0.5 * np.random.randn(2560)

# 발작 감지 함수
def detect_seizure(signal, threshold=1.0):  # 민감도를 높여 threshold를 1.0으로 설정
    seizure_indices = []
    for i in range(len(signal)-1):
        if abs(signal[i+1] - signal[i]) > threshold:  # 급격한 상승 감지
            seizure_indices.append(i)
    return seizure_indices

# 실시간 애니메이션을 위한 준비
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], label="EEG Signal", color='b')

# 발작 구간을 표시할 Rectangle 객체
seizure_patch = patches.Rectangle((0, -2), 0, 4, color='red', alpha=0.5)

ax.add_patch(seizure_patch)

ax.set_xlim(0, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend(loc="upper right")

# 초기화 함수
def init():
    line.set_data([], [])
    seizure_patch.set_width(0)  # 발작이 없으면 길이를 0으로 초기화
    seizure_patch.set_height(0)  # 발작이 없으면 높이를 0으로 초기화
    return line, seizure_patch

# 애니메이션 업데이트 함수
def update(i):
    # 실시간으로 신호 업데이트
    current_data = np.sin(2 * np.pi * 10 * time[:i]) + 0.5 * np.random.randn(i)

    line.set_data(time[:i], current_data)
    
    # 발작 감지
    seizure_indices = detect_seizure(current_data)

    # 발작이 감지되면 빨간색 구간 표시
    if seizure_indices:
        start_time = time[seizure_indices[0]]
        end_time = time[seizure_indices[-1]]
        seizure_patch.set_bounds(start_time, -2, end_time - start_time, 4)  # 위치와 크기 설정
    else:
        seizure_patch.set_width(0)  # 발작이 없으면 길이를 0으로 설정

    return line, seizure_patch

# 애니메이션 실행 (blit=False로 수정)
ani = FuncAnimation(fig, update, frames=range(1, len(time)), init_func=init, blit=False, interval=50)

plt.show()
