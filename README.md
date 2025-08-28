# Anchor Embedding
The official implementation of the paper "Training LLMs to be Better Text Embedders through Bidirectional Reconstruction" (EMNLP 2025 Main Conference).

## Introduction

Large language models (LLMs) have increasingly been explored as powerful text embedders. Existing LLM-based text embedding approaches often leverage the embedding of the final token, typically a reserved special token such as `[EOS]`. However, these tokens have not been intentionally trained to capture the semantics of the whole context, limiting their capacity as text embeddings, especially for retrieval and re-ranking tasks.

We propose to add a new training stage before contrastive learning to enrich the semantics of the final token embedding. This stage employs bidirectional generative reconstruction tasks, namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based Document-to-Query), which interleave to anchor the `[EOS]` embedding and reconstruct either side of Query-Document pairs. 
Experimental results demonstrate that our additional training stage significantly improves LLM performance on the Massive Text Embedding Benchmark (MTEB), achieving new state-of-the-art results across different LLM base models and scales.


![image](https://github.com/LUMIA-Group/Anchor-Embedding/blob/main/method.png)

## Dataset preparation

For both training stages, we use the public portion of dataset used in [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368), curated by authors of [Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449). The dataset can be downloaded from the [GitHub page of Echo embeddings repository](https://github.com/jakespringer/echo-embeddings#training). To use the training script, the downloaded dataset should be placed in the `cache` directory.

## Training

### Stage I

You can simply run training Stage I using the code below:

```bash
torchrun --nproc_per_node=8 experiments/run_stage_I.py train_configs/stage_I/MetaLlama3.2_1b_q2d_d2q.json
```

> [!NOTE]
>
> Our main contribution lies in the `llm2vec/llm2vec_q2d_d2q.py` script, which implements two bidirectional reconstruction tasks — **EBQ2D** and **EBD2Q** — via anchor embeddings.

### Stage II

```bash
torchrun --nproc_per_node=8 experiments/run_stage_II.py train_configs/stage_II/MetaLlama3.2_1b_stage1_2000steps_stage2.json
```

### Baseline

```bash
torchrun --nproc_per_node=8 experiments/run_stage_II.py train_configs/baseline/MetaLlama3.2_1b_baseline_stage2.json
```

## Evaluation

Evaluating on MTEB: 

```bash
python experiments/mteb_eval_custom.py \
  --base_model_name_or_path <path_or_name_of_base_model> \
  --peft_model_name_or_path <path_or_name_of_peft_model> \
  --task_name ${TASK_NAME} \
  --task_to_instructions_fp test_configs/mteb/task_to_instructions.json \
  --output_dir <output_directory>
```

