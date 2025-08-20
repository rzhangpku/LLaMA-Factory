#!/bin/bash

API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api examples/inference/qwen3_lora_sft.yaml
