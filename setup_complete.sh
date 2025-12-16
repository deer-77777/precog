#!/bin/bash
# Complete setup script for LSTM miner

set -e

echo "============================================================"
echo "Complete LSTM Miner Setup"
echo "============================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the precog directory"
    exit 1
fi

echo "Step 1: Installing precog package in development mode..."
pip install -e . || {
    echo "⚠️  Installation with pip install -e failed, trying with --user flag"
    pip install --user -e .
}

echo ""
echo "Step 2: Installing PyTorch..."
pip install torch torchvision || {
    echo "⚠️  PyTorch installation failed, trying with --user flag"
    pip install --user torch torchvision
}

echo ""
echo "Step 3: Verifying installation..."

python3 -c "import precog; print(f'✅ Precog version: {precog.__version__}')" || {
    echo "❌ Failed to import precog"
    exit 1
}

python3 -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')" || {
    echo "❌ Failed to import torch"
    exit 1
}

python3 -c "from precog.miners.models.feature_engineer import FeatureEngineer; print('✅ Can import LSTM modules')" || {
    echo "❌ Failed to import LSTM modules"
    exit 1
}

echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Train model (quick test - 5 minutes):"
echo "   python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10"
echo ""
echo "2. Test predictions:"
echo "   python3 test_local.py"
echo ""
echo "3. Deploy miner:"
echo "   make miner_lstm ENV_FILE=.env.miner"
echo ""

