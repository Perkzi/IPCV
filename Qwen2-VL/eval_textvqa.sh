#!/bin/bash

python textvqa_eval.py \
  --model_name Qwen/Qwen2-VL-7B-Instruct \
  --dataset textvqa \
  --num_samples 1000 \
  --save_samples 200 \
  --output_dir eval_textvqa/results \
  --cache_dir eval_textvqa/data_cache