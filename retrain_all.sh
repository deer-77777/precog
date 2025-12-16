#!/bin/bash
# Retrain all assets (BTC, ETH, TAO)
#
# Usage:
#   ./retrain_all.sh [days] [epochs]
#
# Examples:
#   ./retrain_all.sh          # Use defaults (30 days, 50 epochs)
#   ./retrain_all.sh 60 100   # Use 60 days, 100 epochs

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

# Parameters
DAYS="${1:-30}"    # Default to 30 days
EPOCHS="${2:-50}"  # Default to 50 epochs

# Assets to train
ASSETS=("btc" "eth" "tao_bittensor")

# Set up logging
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/retrain_all_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log "============================================================"
log "RETRAINING ALL ASSETS"
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

# Train each asset
for asset in "${ASSETS[@]}"; do
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Training ${asset^^}..."
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run training
    python3 auto_retrain.py \
        --asset "${asset}" \
        --days "${DAYS}" \
        --epochs "${EPOCHS}" \
        2>&1 | tee -a "${LOG_FILE}"
    
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        log "✅ ${asset^^} training completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log "❌ ${asset^^} training failed with exit code: ${EXIT_CODE}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_ASSETS+=("${asset}")
    fi
    
    # Wait a bit between assets to avoid resource contention
    if [ "${asset}" != "${ASSETS[-1]}" ]; then
        log "Waiting 30 seconds before next asset..."
        sleep 30
    fi
done

log ""
log "============================================================"
log "RETRAINING SUMMARY"
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
    log "🎉 All assets trained successfully!"
fi

log "============================================================"
log ""

# Optionally restart the miner if all succeeded
if [ ${FAIL_COUNT} -eq 0 ]; then
    if command -v pm2 &> /dev/null; then
        log "Checking if miner is running..."
        if pm2 list | grep -q "miner_lstm"; then
            log "Restarting miner to use new models..."
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

