#!/bin/bash
# Wrapper script to run training with correct PYTHONPATH

cd /home/fang/develop/precog
export PYTHONPATH=/home/fang/develop/precog:$PYTHONPATH

python3 precog/miners/scripts/train_lstm.py "$@"

