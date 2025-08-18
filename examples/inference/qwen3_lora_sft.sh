#!/bin/bash

python scripts/vllm_infer.py \
    --dataset alpaca_spoken_test \
    --model_name_or_path /aiplatform-sale/aiplatform-sale/group-shared/models/Qwen/Qwen3-1.7B \
    --adapter_name_or_path saves/Qwen3-1.7B/lora/sft \
    --template qwen3_nothink \
    --finetuning_type lora \
    --infer_backend vllm \
    --trust_remote_code true
