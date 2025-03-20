#!/bin/bash
set -e

export HF_TOKEN=<YOUR_HF_TOKEN_HERE>

judge_model_name=Skywork/Skywork-Critic-Llama-3.1-8B

CUDA_VISIBLE_DEVICES=0 vllm serve ${judge_model_name} \
    --tensor_parallel_size=1 \
    --max_model_len=16384 \
    --gpu_memory_utilization=0.8 \
    --port 8001 \
    --disable-log-stats \
    --disable-log-requests \

