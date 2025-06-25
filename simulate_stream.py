from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# ✅ CORS 설정: Flutter에서 접근 가능하도록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 미리 저장된 EEG 데이터 불러오기 (DWT 완료된)
X = np.load("backend_server/npy/X_dwt.npy")   # shape: (N, 8, 38)
y = np.load("backend_server/npy/y_total.npy") # shape: (N,)
ids = np.load("backend_server/npy/id_list.npy")  # 예: 환자 ID 구분용

# ✅ 인덱스 상태 저장
current_index = {"idx": 0}

# ✅ 스트리밍 응답 타입
class StreamResponse(BaseModel):
    data: list  # EEG chunk [8][38]
    label: int  # 발작 여부 (0/1)

@app.get("/")
def root():
    return {"message": "🧠 EEG stream API is running."}

@app.get("/stream", response_model=StreamResponse)
def stream_sample():
    idx = current_index["idx"]

    if idx >= len(X):
        # 다시 처음부터 반복
        idx = 0
        current_index["idx"] = 0

    # EEG chunk + 라벨 반환
    sample = X[idx]        # shape: [8, 38]
    sample = np.squeeze(sample, axis=0) 
    label = int(y[idx])    # 0 or 1

    print("전체 X shape:", X.shape)
    print("샘플 shape:", sample.shape)


    # 다음 요청을 위해 인덱스 증가
    current_index["idx"] += 1

    return {
        "data": sample.tolist(),  # Python 기본 타입으로 변환해서 반환
        "label": label
    }



# 🟢 **이 부분이 꼭 필요!**
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)



