# KURE: Qwen3-0.6B Anchor Embedding Training

Anchor Embedding 논문(https://arxiv.org/abs/2509.03020) 기반 Qwen3-0.6B Stage I (Q2D/D2Q) 학습 설정.

## Overview

- **Model**: `Qwen/Qwen3-0.6B` (full finetuning, no PEFT)
- **Data**: `/data_x/yjoonjang/PAPERS/Anchor-Embedding/DATA/QP/train_ko_en` (1,862,922 samples, anchor/positive pairs)
- **GPU**: 2x NVIDIA H100 NVL 96GB
- **Loss**: d2q_weight * D2Q + q2d_weight * Q2D (기본값: 0.8/0.2, config에서 조정 가능)

## 생성/수정 파일

### 새로 생성

| 파일 | 설명 |
|------|------|
| `llm2vec/dataset/KoEnQPData.py` | `load_from_disk`으로 anchor/positive 데이터 로드하는 Dataset 클래스 |
| `experiments/run_stage_I_qwen3.py` | Qwen3용 Stage I 학습 스크립트 |
| `train_configs/stage_I/Qwen3_0.6b_q2d_d2q.json` | 학습 설정 파일 |

### 수정

| 파일 | 변경 내용 |
|------|-----------|
| `llm2vec/dataset/__init__.py` | `KoEnQPData` export 추가 |
| `llm2vec/dataset/utils.py` | `"KoEnQP"` 데이터셋 등록 |
| `llm2vec/llm2vec.py` | bidirectional model import → try/except (transformers 5.x 호환) |
| `llm2vec/llm2vec_q2d_d2q.py` | 동일한 import 호환성 수정 |

## 실행 방법

```bash
PYTHONPATH=/mnt/raid6/yjoonjang/projects/Anchor-Embedding \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 \
  experiments/run_stage_I_qwen3.py \
  train_configs/stage_I/Qwen3_0.6b_q2d_d2q.json
```

### WandB 비활성화

```bash
WANDB_DISABLED=true PYTHONPATH=... torchrun ...
```

### stop_after_n_steps 조정

`train_configs/stage_I/Qwen3_0.6b_q2d_d2q.json`에서 `"stop_after_n_steps": 2000` 추가하여 조기 종료 가능. (원 논문 기준 2000 steps 사용)

## 학습 설정 상세

```
per_device_train_batch_size: 32
gradient_accumulation_steps: 2
num_gpus: 2
effective_batch_size: 32 * 2 * 2 = 128

num_train_epochs: 3
total_samples: 1,862,912 (배치 정렬 후)
steps_per_epoch: 1,862,912 / 128 ≈ 14,554
total_steps: ~43,662 (3 epochs)

learning_rate: 4e-5
warmup_steps: 300
max_seq_length: 512
precision: bfloat16
attention: flash_attention_2
gradient_checkpointing: true
```

## Chat Template 결정

Qwen3는 thinking 모델로 `<think>...</think>` 토큰을 사용하지만, Stage I 학습에서는 불필요:

```
적용된 template: <|im_start|>user\n{text}<|im_end|>
```

- **Embedding 추출**: encoder forward pass → `<|im_end|>` 토큰의 hidden state 사용. 생성(generation)이 아니므로 think 토큰 불필요.
- **Reconstruction (D2Q/Q2D)**: teacher forcing으로 정답 토큰을 보며 next-token prediction. 자유 생성이 아니므로 think 토큰 불필요.
- **일관성**: 원본 코드의 Qwen2 template과 동일한 패턴 (`<|im_start|>user\n{text}<|im_end|>`).

EOS 토큰 (`<|im_end|>`, id=151645)이 pad 토큰으로도 사용되며, left padding으로 배치 내 시퀀스 정렬.

## 데이터 형식

원본 데이터 (HuggingFace Dataset):
```
anchor: "서소문본관 합동소방훈련 중 전시동 광장에서는 무슨 훈련을 진행했어"
positive: "서소문본관 합동소방훈련 결과보고 ... 전시동 광장(화재 발생시 소화 훈련)"
```

학습 시 변환:
```
query:    "주어진 질문에 대해 관련 문서를 검색하세요; !@#$%^&*(){anchor}"
positive: "!@#$%^&*(){positive}"
```

`!@#$%^&*()` separator는 instruction과 content를 분리하여 embed_mask를 생성. Pooling 시 instruction 토큰을 제외하고 content 토큰만 사용.

## 출력

체크포인트 저장 위치:
```
output/stage_I/Qwen3-0.6B-q2d-d2q-WO-PEFT/{experiment_id}/
├── checkpoint-{N}/
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── llm2vec_config.json
└── ...
```

## 중간 평가 (NanoBEIR)

학습 중 검색 성능을 NanoBEIR 데이터셋으로 평가. Config에서 `eval_steps`로 주기 설정 (0이면 비활성화).

- **데이터셋**: 4개 (한국어 2개 + 영어 2개)
  - `sionic-ai/NanoBEIR-ko`: NanoMSMARCO, NanoNQ
  - `sentence-transformers/NanoBEIR-en`: NanoMSMARCO, NanoNQ
- **메트릭**: NDCG@10, MRR@10
- **인코딩**: eos_token pooling + cosine similarity
- **로깅**: WandB에 `eval/{lang}/{subset}/ndcg@10`, `eval/{lang}/{subset}/mrr@10`, `eval/avg/ndcg@10`, `eval/avg/mrr@10`
- **실행**: rank 0에서만 실행 (DDP 환경에서 중복 방지)

```json
"eval_steps": 500,
"eval_batch_size": 64
```

Query 인코딩 시 instruction prefix 포함:
```
ko: "주어진 질문에 대해 관련 문서를 검색하세요; !@#$%^&*(){query}"
en: "Retrieve relevant documents for the given query; !@#$%^&*(){query}"
```

Corpus는 instruction 없이 separator만: `"!@#$%^&*(){document}"`

## Stage II로 이어서 학습

Stage I 체크포인트를 Stage II의 `model_name_or_path` 또는 `peft_model_name_or_path`에 지정하여 contrastive learning 수행.
