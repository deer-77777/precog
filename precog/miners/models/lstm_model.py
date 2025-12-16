"""
LSTM Model for Price Prediction

A PyTorch LSTM model for cryptocurrency price prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PricePredictorLSTM(nn.Module):
    """
    LSTM-based model for price prediction.
    
    Architecture:
        Input -> LSTM layers -> Fully Connected -> Output
    
    The model uses:
        - Forget Gate: Decides what information to discard from cell state
        - Input Gate: Decides which new information to store in cell state
        - Output Gate: Decides what to output based on cell state
        - Hidden State: Carries temporal dependencies across time steps
    """
    
    def __init__(
        self,
        input_size: int = 5,  # OHLCV features
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features (default 5 for OHLCV)
            hidden_size: Size of LSTM hidden state
            num_layers: Number of stacked LSTM layers
            output_size: Number of output features (default 1 for price)
            dropout: Dropout rate between LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(PricePredictorLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Fully connected layers
        fc_input_size = hidden_size * self.num_directions
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional tuple of (h_0, c_0) hidden states
        
        Returns:
            output: Predicted values of shape (batch_size, output_size)
            hidden: Tuple of (h_n, c_n) hidden states
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        # c_n shape: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Use the last output for prediction
        # last_output shape: (batch_size, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        return output, (h_n, c_n)
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state with zeros.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            Tuple of (h_0, c_0) initial hidden states
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return (h_0, c_0)
    
    def predict(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Make prediction without returning hidden state.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
        
        Returns:
            Predicted values
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x, hidden)
        return output


class PricePredictorLSTMWithAttention(nn.Module):
    """
    LSTM with Attention mechanism for improved price prediction.
    
    Attention allows the model to focus on the most relevant time steps
    when making predictions.
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super(PricePredictorLSTMWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        
        for module in [self.attention, self.fc]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional hidden state
        
        Returns:
            output: Predicted values
            hidden: Hidden states
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Attention mechanism
        # attention_weights shape: (batch_size, sequence_length, 1)
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        # context shape: (batch_size, hidden_size * num_directions)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        output = self.fc(context)
        
        return output, (h_n, c_n)
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return (h_0, c_0)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction"""
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x)
        return output


def create_model(config: dict, use_attention: bool = False) -> nn.Module:
    """
    Factory function to create LSTM model from config.
    
    Args:
        config: Model configuration dictionary
        use_attention: Whether to use attention mechanism
    
    Returns:
        LSTM model instance
    """
    model_class = PricePredictorLSTMWithAttention if use_attention else PricePredictorLSTM
    
    return model_class(
        input_size=config.get("num_features", 5),
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        output_size=config.get("output_size", 1),
        dropout=config.get("dropout", 0.2),
        bidirectional=config.get("bidirectional", False),
    )

