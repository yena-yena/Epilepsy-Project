import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 더미 데이터 예시: 2초짜리 윈도우 (두 개의 채널)
time = np.linspace(0, 2, 2560)
channel1 = np.sin(2 * np.pi * 10 * time) + 0.5 * np.random.randn(2560)  # 진폭 10Hz
channel2 = np.cos(2 * np.pi * 10 * time) + 0.5 * np.random.randn(2560)  # 진폭 10Hz

fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Channel 1")
line2, = ax.plot([], [], label="Channel 2")
ax.set_xlim(0, 2)  # 시간 (0~2초)
ax.set_ylim(-2, 2)  # 값 범위

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def animate(i):
    x_data = time[:i]
    line1.set_data(x_data, channel1[:i])
    line2.set_data(x_data, channel2[:i])
    return line1, line2

ani = FuncAnimation(fig, animate, frames=len(time), init_func=init, blit=True)
plt.legend()
plt.title("Real-time EEG Signal Simulation")
plt.show()
