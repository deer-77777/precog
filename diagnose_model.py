#!/usr/bin/env python3
"""Diagnose why the model is predicting badly"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from datetime import datetime, timedelta
from precog.utils.cm_data import CMData
from precog.utils.timestamp import to_str
from precog.miners.models.feature_engineer import FeatureEngineer

print("=" * 60)
print("MODEL DIAGNOSIS")
print("=" * 60)

# Load model
model_path = "precog/miners/models/trained_weights/btc_lstm.pth"
checkpoint = torch.load(model_path, map_location='cpu')

print(f"\n1. Model Training Stats:")
print(f"   - Trained epochs: {checkpoint.get('epoch', 'unknown')}")
print(f"   - Validation Loss: {checkpoint.get('val_loss', 'unknown'):.2f}")
print(f"   - Validation MAPE: {checkpoint.get('val_mape', 'unknown'):.4f}%")
print(f"   - Input features: {checkpoint.get('input_size', 'unknown')}")

# Check training data range
print(f"\n2. Fetching recent data...")
cm = CMData()
end_time = datetime.now()
start_time = end_time - timedelta(hours=4)

data = cm.get_CM_ReferenceRate(
    assets=['btc'],
    start=to_str(start_time),
    end=to_str(end_time),
    frequency='1m'
)

current_price = data['ReferenceRateUSD'].iloc[-1]
print(f"   - Current BTC price: ${current_price:,.2f}")
print(f"   - Price range in data: ${data['ReferenceRateUSD'].min():,.2f} - ${data['ReferenceRateUSD'].max():,.2f}")

# Create features
print(f"\n3. Feature Engineering:")
fe = FeatureEngineer(lookback_window=60)
df_features = fe.create_features(data)

feature_cols = checkpoint['feature_cols']
print(f"   - Using {len(feature_cols)} features")

# Check for issues
sequences = fe.prepare_sequences(df_features, feature_cols)
if len(sequences) > 0:
    last_seq = sequences[-1:]
    print(f"   - Sequence shape: {last_seq.shape}")
    print(f"   - Sequence data range: [{last_seq.min():.2f}, {last_seq.max():.2f}]")
    print(f"   - Any NaN: {np.any(np.isnan(last_seq))}")
    print(f"   - Any Inf: {np.any(np.isinf(last_seq))}")
    
    # Check feature scales
    print(f"\n4. Feature Scales:")
    for i, col in enumerate(feature_cols[:5]):
        val = last_seq[0, -1, i]  # Last timestep of last sequence
        print(f"   - {col}: {val:.6f}")

# Make prediction
print(f"\n5. Making Prediction:")
from precog.miners.models.lstm_model import PriceLSTM

model = PriceLSTM(
    input_size=checkpoint['input_size'],
    hidden_size=128,
    num_layers=2
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    sequence_tensor = torch.FloatTensor(last_seq)
    prediction = model(sequence_tensor)
    predicted_price = prediction.cpu().numpy()[0][0]

print(f"   - Model output (raw): {predicted_price:.2f}")
print(f"   - Current price: {current_price:.2f}")
print(f"   - Difference: {abs(predicted_price - current_price):.2f} ({abs(predicted_price - current_price)/current_price*100:.2f}%)")

# Diagnosis
print(f"\n6. Diagnosis:")
if abs(predicted_price - current_price) / current_price > 0.5:
    print("   ❌ CRITICAL: Model prediction is >50% off!")
    print("   Possible causes:")
    print("   1. Training data was too short (7 days is not enough)")
    print("   2. Model didn't converge (try more epochs)")
    print("   3. Learning rate too high/low")
    print("   4. Bad initialization")
    print("")
    print("   SOLUTION: Retrain with:")
    print("   ./retrain_properly.sh")
elif abs(predicted_price - current_price) / current_price > 0.1:
    print("   ⚠️  WARNING: Model prediction is >10% off")
    print("   Consider retraining with more data")
elif abs(predicted_price - current_price) / current_price > 0.05:
    print("   ⚠️  Model is acceptable but could be better")
    print("   MAPE around 5% - Consider retraining with more data")
else:
    print("   ✅ Model prediction looks good!")

print("\n" + "=" * 60)

