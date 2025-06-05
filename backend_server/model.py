import torch
import torch.nn as nn

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
