#!/bin/bash
# Set up automated retraining with cron jobs

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "Setting Up Automated Retraining"
echo "============================================================"
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Make scripts executable
chmod +x "${PROJECT_ROOT}/retrain_schedule.sh"
chmod +x "${PROJECT_ROOT}/auto_retrain.py"
chmod +x "${PROJECT_ROOT}/incremental_train.py"

echo "✅ Made scripts executable"
echo ""

# Show current crontab
echo "Current cron jobs:"
echo "----------------------------------------"
crontab -l 2>/dev/null || echo "(No cron jobs configured yet)"
echo "----------------------------------------"
echo ""

# Ask user which assets to train
echo "Which assets do you want to auto-retrain?"
echo ""
echo "1. BTC only"
echo "2. BTC + ETH"
echo "3. BTC + ETH + TAO (all assets)"
echo ""
read -p "Enter choice [1-3]: " asset_choice

case $asset_choice in
    1)
        ASSETS=("btc")
        ASSET_DESC="BTC"
        ;;
    2)
        ASSETS=("btc" "eth")
        ASSET_DESC="BTC + ETH"
        ;;
    3)
        ASSETS=("btc" "eth" "tao_bittensor")
        ASSET_DESC="BTC + ETH + TAO"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Selected assets: ${ASSET_DESC}"
echo ""

# Ask user which schedule to use
echo "Choose retraining schedule:"
echo ""
echo "1. Conservative  - Full retrain every 7 days (recommended for beginners)"
echo "2. Moderate      - Full retrain every 3 days (recommended)"
echo "3. Aggressive    - Full retrain every 2 days + daily fine-tuning (advanced)"
echo "4. Manual        - Don't set up automatic retraining (you'll run manually)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        # Conservative: Every 7 days at 2am, staggered by asset
        SCHEDULE="Every 7 days at 2:00 AM (staggered)"
        ;;
    2)
        # Moderate: Every 3 days at 2am, staggered by asset
        SCHEDULE="Every 3 days at 2:00 AM (staggered)"
        ;;
    3)
        # Aggressive: Every 2 days full + daily incremental
        SCHEDULE="Full: Every 2 days at 2:00 AM\n        Incremental: Daily at 3:00 AM"
        ;;
    4)
        echo ""
        echo "✅ Manual mode selected"
        echo ""
        echo "To retrain manually, run:"
        echo "  cd ${PROJECT_ROOT}"
        echo "  ./retrain_all.sh    # Train all assets"
        echo "  # Or train individually:"
        echo "  ./retrain_schedule.sh btc 30 50"
        echo "  ./retrain_schedule.sh eth 30 50"
        echo "  ./retrain_schedule.sh tao_bittensor 30 50"
        echo ""
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Add to crontab
echo ""
echo "Adding cron job(s)..."
echo "Schedule: ${SCHEDULE}"
echo ""

# Get current crontab
crontab -l 2>/dev/null > /tmp/current_cron || touch /tmp/current_cron

# Add new job(s) for each asset
case $choice in
    1)
        # Conservative: Every 7 days, staggered by 2 hours per asset
        hour=2
        for asset in "${ASSETS[@]}"; do
            echo "0 ${hour} */7 * * ${PROJECT_ROOT}/retrain_schedule.sh ${asset} 30 50 >> ${PROJECT_ROOT}/logs/cron.log 2>&1" >> /tmp/current_cron
            hour=$((hour + 2))
        done
        ;;
    2)
        # Moderate: Every 3 days, staggered by 1 hour per asset
        hour=2
        for asset in "${ASSETS[@]}"; do
            echo "0 ${hour} */3 * * ${PROJECT_ROOT}/retrain_schedule.sh ${asset} 30 50 >> ${PROJECT_ROOT}/logs/cron.log 2>&1" >> /tmp/current_cron
            hour=$((hour + 1))
        done
        ;;
    3)
        # Aggressive: Every 2 days full + daily incremental, staggered
        # Full retraining
        hour=2
        for asset in "${ASSETS[@]}"; do
            echo "0 ${hour} */2 * * ${PROJECT_ROOT}/retrain_schedule.sh ${asset} 30 50 >> ${PROJECT_ROOT}/logs/cron.log 2>&1" >> /tmp/current_cron
            hour=$((hour + 1))
        done
        # Incremental fine-tuning (3 hours after full training starts)
        hour=5
        for asset in "${ASSETS[@]}"; do
            echo "0 ${hour} * * * ${PROJECT_ROOT}/incremental_train.py --asset ${asset} --days 7 --epochs 10 >> ${PROJECT_ROOT}/logs/cron.log 2>&1" >> /tmp/current_cron
            hour=$((hour + 1))
        done
        ;;
esac

# Install new crontab
crontab /tmp/current_cron
rm /tmp/current_cron

echo "✅ Cron jobs configured successfully!"
echo ""

# Show new crontab
echo "New cron configuration:"
echo "----------------------------------------"
crontab -l
echo "----------------------------------------"
echo ""

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "📊 Assets: ${ASSET_DESC}"
echo "📅 Schedule: ${SCHEDULE}"
echo ""
echo "📊 Logs will be saved to: ${PROJECT_ROOT}/logs/"
echo ""
echo "To view recent logs:"
echo "  tail -f ${PROJECT_ROOT}/logs/auto_retrain_\$(date +%Y%m%d).log"
echo ""
echo "To check cron logs:"
echo "  tail -f ${PROJECT_ROOT}/logs/cron.log"
echo ""
echo "To manually trigger retraining now:"
echo "  ${PROJECT_ROOT}/retrain_all.sh    # Train all assets"
echo "  # Or train individually:"
for asset in "${ASSETS[@]}"; do
    echo "  ${PROJECT_ROOT}/retrain_schedule.sh ${asset} 30 50"
done
echo ""
echo "To remove cron jobs:"
echo "  crontab -e  # then delete the lines"
echo ""
echo "✅ Your miner will now automatically update models for all selected assets!"
echo ""

