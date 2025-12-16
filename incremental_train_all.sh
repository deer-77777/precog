#!/bin/bash
# Incremental fine-tuning for all assets (BTC, ETH, TAO)
#
# Usage:
#   ./incremental_train_all.sh [days] [epochs]
#
# Examples:
#   ./incremental_train_all.sh          # Use defaults (7 days, 10 epochs)
#   ./incremental_train_all.sh 14 20    # Use 14 days, 20 epochs

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

# Parameters
DAYS="${1:-7}"     # Default to 7 days
EPOCHS="${2:-10}"  # Default to 10 epochs

# Assets to train
ASSETS=("btc" "eth" "tao_bittensor")

# Set up logging
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/incremental_all_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log "============================================================"
log "INCREMENTAL TRAINING - ALL ASSETS"
log "============================================================"
log "Days of data: ${DAYS}"
log "Epochs: ${EPOCHS}"
log "Assets: ${ASSETS[*]}"
log "============================================================"
log ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    log "Activating virtual environment..."
    source venv/bin/activate
fi

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_ASSETS=()

# Fine-tune each asset
for asset in "${ASSETS[@]}"; do
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Fine-tuning ${asset^^}..."
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run incremental training
    python3 incremental_train.py \
        --asset "${asset}" \
        --days "${DAYS}" \
        --epochs "${EPOCHS}" \
        --learning-rate 0.0001 \
        2>&1 | tee -a "${LOG_FILE}"
    
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        log "✅ ${asset^^} fine-tuning completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log "❌ ${asset^^} fine-tuning failed with exit code: ${EXIT_CODE}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_ASSETS+=("${asset}")
    fi
    
    # Wait a bit between assets
    if [ "${asset}" != "${ASSETS[-1]}" ]; then
        log "Waiting 10 seconds before next asset..."
        sleep 10
    fi
done

log ""
log "============================================================"
log "INCREMENTAL TRAINING SUMMARY"
log "============================================================"
log "Total assets: ${#ASSETS[@]}"
log "✅ Successful: ${SUCCESS_COUNT}"
log "❌ Failed: ${FAIL_COUNT}"

if [ ${FAIL_COUNT} -gt 0 ]; then
    log ""
    log "Failed assets:"
    for asset in "${FAILED_ASSETS[@]}"; do
        log "  - ${asset}"
    done
    log ""
    log "⚠️ Some assets failed. Check logs for details."
    log "Log file: ${LOG_FILE}"
else
    log ""
    log "🎉 All assets fine-tuned successfully!"
fi

log "============================================================"
log ""

# Optionally restart the miner if all succeeded
if [ ${FAIL_COUNT} -eq 0 ]; then
    if command -v pm2 &> /dev/null; then
        log "Checking if miner is running..."
        if pm2 list | grep -q "miner_lstm"; then
            log "Restarting miner to use updated models..."
            pm2 restart miner_lstm 2>&1 | tee -a "${LOG_FILE}"
            
            if [ $? -eq 0 ]; then
                log "✅ Miner restarted successfully"
            else
                log "⚠️ Failed to restart miner (please check manually)"
            fi
        else
            log "ℹ️ Miner not running (no restart needed)"
        fi
    fi
fi

log "Complete log saved to: ${LOG_FILE}"

# Exit with failure if any asset failed
exit ${FAIL_COUNT}

