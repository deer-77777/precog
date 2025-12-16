#!/bin/bash
# Train production-quality LSTM model

echo "============================================================"
echo "PRODUCTION LSTM TRAINING"
echo "============================================================"
echo ""
echo "This will train a high-quality model with:"
echo "  - 60 days of historical data (vs 7 days)"
echo "  - 100 training epochs (vs 10 epochs)"
echo "  - Better hyperparameters"
echo ""
echo "Expected results:"
echo "  - MAPE: 0.5-2% (currently 52%)"
echo "  - Prediction error: < $70 (currently $196)"
echo "  - Interval width: ~$250"
echo ""
echo "This will take ~1 hour"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    cd /home/fang/develop/precog
    
    # Remove old bad model
    echo ""
    echo "Removing old model..."
    rm -f precog/miners/models/trained_weights/btc_lstm.pth
    
    # Test Binance API first
    echo ""
    echo "Testing Binance API..."
    python3 test_binance.py
    
    echo ""
    echo "============================================================"
    echo "Starting production training..."
    echo "============================================================"
    
    # Train with optimal parameters
    python3 precog/miners/scripts/train_lstm.py \
        --assets btc \
        --days 60 \
        --epochs 100 \
        --batch_size 64 \
        --learning_rate 0.00005 \
        --use-binance
    
    echo ""
    echo "============================================================"
    echo "Training complete!"
    echo "============================================================"
    echo ""
    echo "Test the model:"
    echo "  python3 test_local.py"
    echo ""
    echo "Expected results:"
    echo "  - MAPE < 2%"
    echo "  - Prediction error < $70"
    echo "  - Interval width ~$250"
    echo ""
else
    echo "Cancelled"
fi

