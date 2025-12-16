#!/bin/bash
# Setup script for LSTM miner

set -e

echo "============================================================"
echo "LSTM Miner Setup"
echo "============================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the precog directory"
    exit 1
fi

echo "1. Installing PyTorch..."
pip install torch torchvision 2>&1 | grep -v "already satisfied" || echo "   ✓ PyTorch already installed"

echo ""
echo "2. Verifying installation..."
python3 -c "import torch; print('   ✓ PyTorch version:', torch.__version__)"

echo ""
echo "3. Checking directory structure..."
mkdir -p precog/miners/models/trained_weights
mkdir -p precog/miners/scripts
echo "   ✓ Directories created"

echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Train models (this will take some time):"
echo "   python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10"
echo ""
echo "2. Test predictions:"
echo "   python3 precog/miners/scripts/test_lstm.py"
echo ""
echo "3. Deploy miner:"
echo "   Update .env.miner: FORWARD_FUNCTION=lstm_miner"
echo "   make miner_lstm ENV_FILE=.env.miner"
echo ""

