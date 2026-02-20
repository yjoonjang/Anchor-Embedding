export WANDB_PROJECT="KURE-v2-Q2D-D2Q"


PYTHONPATH=/mnt/raid6/yjoonjang/projects/Anchor-Embedding \
CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun --nproc_per_node=2 \
	experiments/run_stage_I_qwen3.py \
	train_configs/stage_I/Qwen3_0.6b_q2d_d2q.json