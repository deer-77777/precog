"""LSTM models for cryptocurrency price prediction"""

import torch
import torch.nn as nn


class PriceLSTM(nn.Module):
    """LSTM model for cryptocurrency price prediction"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Number of features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(PriceLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)  # Output: single price prediction

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predicted price (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class PriceIntervalLSTM(nn.Module):
    """LSTM model with dual outputs: point prediction + interval bounds"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(PriceIntervalLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Shared layers
        self.fc_shared = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Point prediction head
        self.fc_point = nn.Linear(64, 1)

        # Interval prediction heads
        self.fc_lower = nn.Linear(64, 1)
        self.fc_upper = nn.Linear(64, 1)

    def forward(self, x):
        """
        Returns:
            point: Point prediction
            lower: Lower bound of interval
            upper: Upper bound of interval
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        # Shared representation
        shared = self.fc_shared(last_output)
        shared = self.relu(shared)
        shared = self.dropout(shared)

        # Point prediction
        point = self.fc_point(shared)

        # Interval bounds
        lower = self.fc_lower(shared)
        upper = self.fc_upper(shared)

        return point, lower, upper

