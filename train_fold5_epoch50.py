import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("saved_models", exist_ok=True)

class CNNBiLSTMModel(nn.Module):
    def __init__(self, input_channels, input_time):
        super(CNNBiLSTMModel, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(32 * 2, 1)

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(1)
        if x.shape[1] != self.input_channels:
            x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def calculate_pos_weight(y_train):
    pos = np.sum(y_train)
    neg = len(y_train) - pos
    weight = neg / (pos + 1e-5)
    return weight

def train_k_fold(data_by_patient, k=5):
    patient_ids = list(data_by_patient.keys())
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(patient_ids)):
        print(f"===== Fold {fold + 1} =====")
        train_ids = [patient_ids[i] for i in train_index]
        val_ids = [patient_ids[i] for i in val_index]

        X_train = np.concatenate([data_by_patient[pid]['X'] for pid in train_ids])
        y_train = np.concatenate([data_by_patient[pid]['y'] for pid in train_ids])
        X_val = np.concatenate([data_by_patient[pid]['X'] for pid in val_ids])
        y_val = np.concatenate([data_by_patient[pid]['y'] for pid in val_ids])

        print(f"[DEBUG] y_train 발작 비율: {np.mean(y_train):.4f}, y_val 발작 비율: {np.mean(y_val):.4f}")

        if len(np.unique(y_val)) < 2:
            print(f"⚠️ Fold {fold+1}: validation set has only one class. Skipping this fold.")
            continue

        pos_weight_val = calculate_pos_weight(y_train)
        print(f"[INFO] pos_weight 자동 계산값: {pos_weight_val:.2f}")
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)

        model = CNNBiLSTMModel(input_channels=8, input_time=38).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        best_f1 = 0.0
        for epoch in range(50):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                if X_batch.ndim == 4:
                    X_batch = X_batch.squeeze(1)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

            # 평가 및 성능 향상 시 저장
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    if X_batch.ndim == 4:
                        X_batch = X_batch.squeeze(1)
                    outputs = torch.sigmoid(model(X_batch).squeeze())
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(y_batch.numpy())

            preds_bin = [1 if p > 0.5 else 0 for p in all_preds]

            auc = roc_auc_score(all_labels, all_preds)
            f1 = f1_score(all_labels, preds_bin)
            if f1 > best_f1:
                best_f1 = f1
                save_path = f"backend_server/saved_models/model_fold{fold+1}_best.pt"
                torch.save(model.state_dict(), save_path)
                print(f"✅ Best model updated and saved to {save_path} (F1: {f1:.4f})")

        print(f"Final Fold {fold+1} Accuracy: {accuracy_score(all_labels, preds_bin):.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    X = np.load("backend_server/npy/X_dwt.npy")
    y = np.load("backend_server/npy/y_total.npy")
    ids = np.load("backend_server/npy/id_list.npy")

    data_by_patient = defaultdict(lambda: {'X': [], 'y': []})
    for i in range(len(X)):
        pid = ids[i]
        data_by_patient[pid]['X'].append(X[i])
        data_by_patient[pid]['y'].append(y[i])

    for pid in data_by_patient:
        data_by_patient[pid]['X'] = np.stack(data_by_patient[pid]['X'])
        data_by_patient[pid]['y'] = np.array(data_by_patient[pid]['y'])

    train_k_fold(data_by_patient)
