#!/bin/bash

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
