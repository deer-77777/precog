#!/bin/bash
# Script to retrain model with better parameters

echo "============================================================"
echo "⚠️  WARNING: Your current model has 52% error!"
echo "This script will retrain with better settings"
echo "============================================================"
echo ""

cd /home/fang/develop/precog

# Remove bad model
echo "Removing old model..."
rm -f precog/miners/models/trained_weights/btc_lstm.pth

echo ""
echo "Retraining with better parameters..."
echo "- Data source: Binance API (free, unlimited)"
echo "- More data: 30 days (instead of 7)"
echo "- More epochs: 50 (instead of 10)"
echo "- This will take ~30 minutes"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Test Binance API first
    echo ""
    echo "Testing Binance API..."
    python3 test_binance.py
    
    echo ""
    echo "Starting training..."
    python3 precog/miners/scripts/train_lstm.py \
        --assets btc \
        --days 30 \
        --epochs 50 \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --use-binance
    
    echo ""
    echo "============================================================"
    echo "Retraining complete! Test again:"
    echo "  python3 test_local.py"
    echo "============================================================"
else
    echo "Cancelled"
fi

