#!/bin/bash

#export INFERENCE_MODEL="/mnt/gpt-storage/models/lotus-12B-fixed"
export INFERENCE_MODEL="./models/lotus-12B-fixed"
export INFERENCE_PORT="8888"
python3 inference.py
