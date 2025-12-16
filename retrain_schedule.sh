#!/bin/bash
# Scheduled Retraining Script
# 
# This script is designed to run via cron job
# It handles errors gracefully and logs everything

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

# Set up logging
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/auto_retrain_$(date +%Y%m%d).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log "============================================================"
log "SCHEDULED RETRAINING STARTED"
log "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    log "❌ ERROR: python3 not found"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    log "Activating virtual environment..."
    source venv/bin/activate
fi

# Asset to retrain (can be changed)
ASSET="${1:-btc}"  # Default to BTC if no argument
DAYS="${2:-30}"    # Default to 30 days
EPOCHS="${3:-50}"  # Default to 50 epochs

log "Asset: ${ASSET}"
log "Training days: ${DAYS}"
log "Epochs: ${EPOCHS}"

# Run retraining
log "Starting automated retraining..."
python3 auto_retrain.py \
    --asset "${ASSET}" \
    --days "${DAYS}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee -a "${LOG_FILE}"

RETRAIN_EXIT_CODE=$?

if [ ${RETRAIN_EXIT_CODE} -eq 0 ]; then
    log "✅ Retraining completed successfully"
    
    # Optionally restart the miner if it's running
    if command -v pm2 &> /dev/null; then
        log "Checking if miner is running..."
        if pm2 list | grep -q "miner_lstm"; then
            log "Restarting miner to use new model..."
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
    
    # Send success notification (optional)
    # Uncomment and configure if you want email/slack notifications
    # curl -X POST https://your-webhook-url \
    #     -d "Retraining successful for ${ASSET}"
    
else
    log "❌ Retraining failed with exit code: ${RETRAIN_EXIT_CODE}"
    
    # Send failure notification (optional)
    # curl -X POST https://your-webhook-url \
    #     -d "⚠️ Retraining FAILED for ${ASSET}"
fi

log "============================================================"
log "SCHEDULED RETRAINING FINISHED"
log "Exit code: ${RETRAIN_EXIT_CODE}"
log "============================================================"
log ""

# Clean up old logs (keep last 30 days)
find "${LOG_DIR}" -name "auto_retrain_*.log" -mtime +30 -delete

exit ${RETRAIN_EXIT_CODE}

