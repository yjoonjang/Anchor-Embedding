WANDB_PROJECT="KURE-v2-Q2D-D2Q"

# Stage I checkpoint path
STAGE1_CKPT="/data_x/yjoonjang/PAPERS/Anchor-Embedding/MODELS/stage1/KoEnQP_train_m-Qwen3-0.6B_g-2_b-16_ga-4_l-2048_e-1_w-0_lr-0.0001_d2q-0.5_q2d-0.5/checkpoint-14554"

OUTPUT_DIR="/data_x/yjoonjang/PAPERS/Anchor-Embedding/MODELS/stage2"
TRAIN_DATA="/data_x/yjoonjang/PAPERS/Anchor-Embedding/DATA/SFT/train_ko_en_filtered"
EVAL_DATA="yjoonjang/markers_bm"
MAX_SEQ_LENGTH=2048
NUM_EPOCHS=1	
LEARNING_RATE=2e-5
PER_DEVICE_TRAIN_BATCH_SIZE=2048
MINI_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_RATIO=0.1
TEMPERATURE=0.01
MARGIN_STRATEGY="absolute"
MARGIN=-0.1
HARDNESS_ALPHA=5.0
HARDNESS_MODE="hard_only"
EVAL_ON_START=True
SAVE_STEPS=0.2
EVAL_STEPS=0.05
LOGGING_STEPS=2
BF16=True
GRADIENT_CHECKPOINTING=False
SEED=42
WANDB_PROJECT="KURE-v2-Q2D-D2Q"
RUN_NAME="stage2_GIST_${MARGIN_STRATEGY}_m${MARGIN}_t${TEMPERATURE}_w${WARMUP_RATIO}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_ha${HARDNESS_ALPHA}_${HARDNESS_MODE}"


PYTHONPATH=/mnt/raid6/yjoonjang/projects/Anchor-Embedding \
CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun --nproc_per_node=2 \
	stage2_SFT/train.py \
	--model_name_or_path="${STAGE1_CKPT}" \
	--output_dir="${OUTPUT_DIR}" \
	--train_data="${TRAIN_DATA}" \
	--eval_data="${EVAL_DATA}" \
	--max_seq_length="${MAX_SEQ_LENGTH}" \
	--num_epochs="${NUM_EPOCHS}" \
	--learning_rate="${LEARNING_RATE}" \
	--per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
	--mini_batch_size="${MINI_BATCH_SIZE}" \
	--per_device_eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE}" \
	--gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
	--warmup_ratio="${WARMUP_RATIO}" \
	--temperature="${TEMPERATURE}" \
	--margin_strategy="${MARGIN_STRATEGY}" \
	--margin="${MARGIN}" \
	--hardness_alpha="${HARDNESS_ALPHA}" \
	--hardness_mode="${HARDNESS_MODE}" \
	--eval_on_start="${EVAL_ON_START}" \
	--save_steps="${SAVE_STEPS}" \
	--eval_steps="${EVAL_STEPS}" \
	--logging_steps="${LOGGING_STEPS}" \
	--bf16="${BF16}" \
	--gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
	--seed="${SEED}" \
	--wandb_project="${WANDB_PROJECT}" \
	--run_name="${RUN_NAME}"
