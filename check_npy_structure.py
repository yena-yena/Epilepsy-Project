import numpy as np

data = np.load("X_dwt_all.npy", allow_pickle = True)
print(type(data))

try:
    print("키 목록 : ", data.keys())

except:
    print("배열 shape : ", data.shape)
    print("예시 데이터 : ", data[0])