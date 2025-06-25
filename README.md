# 🧠 Epilepsy Project

A machine learning-based system to detect precursor symptoms of epileptic seizures using EEG data in real time.

![GitHub repo size](https://img.shields.io/github/repo-size/yena-yena/Epilepsy-Project)
![GitHub contributors](https://img.shields.io/github/contributors/yena-yena/Epilepsy-Project)
![GitHub license](https://img.shields.io/github/license/yena-yena/Epilepsy-Project)

---

## 🚀 Project Overview

This project aims to develop a **real-time seizure precursor detection system** using EEG signals.  
By leveraging deep learning models such as **CNN** and **Bi-LSTM**, the system predicts the likelihood of epileptic seizures based on EEG waveforms.

### 🔹 Key Features

- ✅ **Deep Learning Architecture**: Combined **1D CNN + Bi-LSTM** model for temporal EEG signal analysis  
- ✅ **Real-time EEG Estimator**: Efficient, lightweight, and compatible with mobile devices  
- ✅ **Seizure Alert System**: Provides early warnings to help prevent accidents (e.g., falls, head trauma)  

---

## 📦 Model Download

> The trained model used for inference (`model_fold1_best.pt`) is available at the link below:

🔗 [Download model_fold1_best.pt from Google Drive](https://drive.google.com/uc?id=링크ID)

> Please make sure to place it in the following directory before running the server:  
`/backend_server/saved_models/model_fold1_best.pt`

---

## 🧪 Inference Server (FastAPI)

To run the backend server:

```bash
cd backend_server
uvicorn main:app --reload
