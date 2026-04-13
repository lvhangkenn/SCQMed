#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VISIBLE_DEVICES='0' python "$SCRIPT_DIR/../main_new_MM.py" \
  --device cuda:0 \
  --data_path_1 ./data/MIMIC3/proc_text_embedding.pkl \
  --data_path_2 ./data/MIMIC3/proc_kg_embedding.pkl \
  --ckpt_dir ./checkpoint/ \
  --e_dim 64 \
  --num_emb_list 256 256 256 256 \
  --layers 1024 512 256 128 64 \
  --lr 1e-3 \
  --batch_size 1024 \
  --maxe 2000 \
  --loss_type mse \
  --recon 3