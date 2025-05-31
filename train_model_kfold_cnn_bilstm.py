import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import defaultdict
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN1DModel(nn.Module):
    def __init__(self, input_length):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * (input_length // 2), 1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
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

def find_best_threshold(y_true, y_probs):
    best_f1 = 0
    best_th = 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (y_probs > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = t
    return best_th, best_f1

def train_k_fold(data_by_patient, n_splits=8, epochs=50, batch_size=32):
    patient_ids = list(data_by_patient.keys())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_fold_acc = []
    all_fold_auc = []

    fold = 1
    for train_idx, val_idx in kf.split(patient_ids):
        print(f'\n===== Fold {fold} =====')
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]

        X_train, y_train = [], []
        for pid in train_patients:
            X, y = data_by_patient[pid]
            X_train.append(X)
            y_train.append(y)
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train).astype(int)

        if len(np.unique(y_train)) < 2:
            print(f"⚠️ fold {fold}에는 클래스가 하나뿐이야! SMOTE 못 씀.")
            continue

        # Reshape for SMOTE
        X_train_seq = X_train.squeeze(1)  # (N, 8, 38)
        n_samples, n_channels, n_time = X_train_seq.shape
        X_train_flat = X_train_seq.reshape(n_samples, -1)  # (N, 304)

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)

        # Reshape back to (N, 1, 304) for CNN1D
        X_resampled = X_resampled.reshape(-1, 1, n_channels * n_time)

        X_val, y_val = [], []
        for pid in val_patients:
            X, y = data_by_patient[pid]
            X_val.append(X)
            y_val.append(y)
        X_val = np.concatenate(X_val).reshape(-1, 1, n_channels * n_time)
        y_val = np.concatenate(y_val)

        pos_weight = torch.tensor(3.0, dtype=torch.float32).to(DEVICE)
        model = CNN1DModel(input_length=n_channels * n_time).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_loader = DataLoader(EEGDataset(X_resampled, y_resampled), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size)

        train_losses = []
        val_aucs = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_losses.append(total_loss / len(train_loader))

            model.eval()
            y_true_val, y_pred_val = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    outputs = model(X_batch).squeeze()
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    y_true_val.extend(y_batch.numpy())
                    y_pred_val.extend(probs)

            y_true_val = np.array(y_true_val)
            y_pred_val = np.array(y_pred_val)
            best_th, best_f1 = find_best_threshold(y_true_val, y_pred_val)
            y_pred_binary = (y_pred_val > best_th).astype(int)

            val_auc = roc_auc_score(y_true_val, y_pred_val)
            val_acc = accuracy_score(y_true_val, y_pred_binary)
            precision = precision_score(y_true_val, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_val, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_val, y_pred_binary, zero_division=0)
            val_aucs.append(val_auc)

            print(f"\U0001F4CA Epoch {epoch+1}/{epochs} (th={best_th:.2f}) - AUC: {val_auc:.4f}, ACC: {val_acc:.4f}, F1: {f1:.4f}")

        all_fold_acc.append(val_acc)
        all_fold_auc.append(val_auc)

        plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
        plt.plot(range(1, epochs+1), val_aucs, label="Val AUC")
        plt.title(f"Fold {fold} Loss & AUC")
        plt.legend()
        plt.grid()
        plt.show()

        fold += 1

    print("\n==== 전체 평균 성능 ====")
    print("평균 Accuracy:", np.mean(all_fold_acc))
    print("평균 AUC:", np.mean(all_fold_auc))

if __name__ == "__main__":
    X = np.load("npy/X_dwt.npy")
    y = np.load("npy/y_total.npy")
    ids = np.load("npy/id_list.npy")

    data_by_patient = defaultdict(lambda: [[], []])
    for i in range(len(X)):
        pid = ids[i]
        data_by_patient[pid][0].append(X[i])
        data_by_patient[pid][1].append(y[i])

    for pid in data_by_patient:
        data_by_patient[pid] = (
            np.stack(data_by_patient[pid][0]),
            np.array(data_by_patient[pid][1])
        )

    train_k_fold(data_by_patient)
