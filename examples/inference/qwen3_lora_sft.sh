#!/bin/bash

start=$(date +%s.%N)

python scripts/vllm_infer.py \
    --dataset alpaca_spoken_test \
    --model_name_or_path /aiplatform-sale/aiplatform-sale/group-shared/models/Qwen/Qwen3-1.7B \
    --adapter_name_or_path saves/Qwen3-1.7B/lora/sft \
    --template qwen3_nothink \
    --enable_thinking false \
    --batch_size 1

end=$(date +%s.%N)

runtime=$(echo "$end - $start" | bc)
echo "运行时间: $runtime 秒"
