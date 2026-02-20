"""Stage II SFT: Fine-tune Stage I checkpoint with CachedGISTEmbedLoss.

Loads a Stage I pre-trained model (e.g., Qwen3-0.6B) with EOS (last token) pooling
using sentence-transformers, and fine-tunes with guided contrastive learning.

The tokenizer is configured to auto-append the EOS token (<|im_end|>) via
TemplateProcessing (ref: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/discussions/2),
so query prompts are handled cleanly via sentence-transformers' `prompts` feature.

Usage:
	python stage2_SFT/train.py --model_name_or_path /path/to/stage1/checkpoint ...
	# Or via torchrun for distributed training:
	torchrun --nproc_per_node=2 stage2_SFT/train.py --model_name_or_path ...
"""

import logging
import os

import fire
import torch
from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from setproctitle import setproctitle
from tokenizers.processors import TemplateProcessing

from stage2_SFT.losses import CachedGISTEmbedLoss

logging.basicConfig(
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)
logger = logging.getLogger(__name__)

QUERY_PROMPT = (
	"Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: "
)


def build_model(
	model_name_or_path: str,
	max_seq_length: int = 512,
	attn_implementation: str = "flash_attention_2",
) -> SentenceTransformer:
	"""Build SentenceTransformer from a HuggingFace checkpoint with lasttoken pooling.

	1. Loads the base transformer (Qwen3Model) from the Stage I checkpoint.
	2. Configures left padding for natural last-token extraction.
	3. Adds TemplateProcessing so the tokenizer auto-appends the EOS token.
	4. Wraps with Pooling(lasttoken) for EOS embedding extraction.
	"""
	word_embedding_model = Transformer(
		model_name_or_path,
		max_seq_length=max_seq_length,
		model_args={
			"attn_implementation": attn_implementation,
			"dtype": torch.bfloat16,
		},
		tokenizer_args={"padding_side": "left"},
	)

	# Configure tokenizer to auto-append EOS token (<|im_end|>)
	# Ref: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/discussions/2
	tokenizer = word_embedding_model.tokenizer
	eos_token = tokenizer.eos_token
	eos_token_id = tokenizer.eos_token_id
	tokenizer._tokenizer.post_processor = TemplateProcessing(
		single=f"$A {eos_token}:0",
		pair=f"$A {eos_token}:0 $B:1 {eos_token}:1",
		special_tokens=[(eos_token, eos_token_id)],
	)
	logger.info(f"Tokenizer configured to auto-append EOS: {eos_token} (id={eos_token_id})")

	pooling_model = Pooling(
		word_embedding_model.get_word_embedding_dimension(),
		pooling_mode="lasttoken",
	)

	model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
	model.max_seq_length = max_seq_length
	return model


def load_eval_data(eval_data: str, query_prompt: str = QUERY_PROMPT):
	"""Load IR evaluation dataset and create evaluator."""
	dev_corpus = hf_load_dataset(eval_data, "corpus", split="corpus")
	dev_queries = hf_load_dataset(eval_data, "queries", split="queries")

	split = "test" if eval_data == "yjoonjang/markers_bm" else "dev"
	relevant_docs_data = hf_load_dataset(eval_data, "default", split=split)

	queries = {str(q["_id"]): q["text"] for q in dev_queries}
	corpus = {str(c["_id"]): c["text"] for c in dev_corpus}

	relevant_docs = {}
	for qid, corpus_id in zip(
		relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]
	):
		qid, corpus_id = str(qid), str(corpus_id)
		if qid not in relevant_docs:
			relevant_docs[qid] = set()
		relevant_docs[qid].add(corpus_id)

	return InformationRetrievalEvaluator(
		queries=queries,
		corpus=corpus,
		relevant_docs=relevant_docs,
		query_prompt=query_prompt,
		write_csv=False,
		name=eval_data.replace("/", "_"),
	)


def train(
	# Model
	model_name_or_path: str = "/data_x/yjoonjang/PAPERS/Anchor-Embedding/MODELS/stage1/KoEnQP_train_m-Qwen3-0.6B_g-2_b-16_ga-4_l-2048_e-1_w-0_lr-0.0001_d2q-0.5_q2d-0.5/checkpoint-14554",
	guide_model_name_or_path: str = None,
	attn_implementation: str = "flash_attention_2",
	max_seq_length: int = 512,
	# Data
	train_data: str = "/data_x/yjoonjang/PAPERS/Anchor-Embedding/DATA/QP/train_ko_en",
	eval_data: str = "yjoonjang/markers_bm",
	query_prompt: str = QUERY_PROMPT,
	# Loss
	mini_batch_size: int = 32,
	temperature: float = 0.01,
	margin_strategy: str = "absolute",
	margin: float = -0.1,
	hardness_alpha: float = 0.0,
	hardness_mode: str = "all",
	# Training
	output_dir: str = "/data_x/yjoonjang/PAPERS/Anchor-Embedding/MODELS/stage2",
	num_epochs: int = 1,
	learning_rate: float = 2e-5,
	per_device_train_batch_size: int = 512,
	per_device_eval_batch_size: int = 256,
	gradient_accumulation_steps: int = 1,
	warmup_ratio: float = 0.1,
	logging_steps: int = 2,
	save_strategy: str = "steps",
	save_steps: int = 500,
	eval_strategy: str = "steps",
	eval_steps: int = 500,
	bf16: bool = True,
	seed: int = 42,
	gradient_checkpointing: bool = False,
	eval_on_start: bool = False,
	dataloader_num_workers: int = 4,
	# Wandb
	use_wandb: bool = True,
	wandb_project: str = "Anchor-Embedding-Stage2",
	run_name: str = None,
	# Resume
	resume_from_checkpoint: str = None,
):
	"""Stage II SFT with CachedGISTEmbedLoss (self-guided or with separate guide).

	Loads a Stage I pre-trained checkpoint with EOS (lasttoken) pooling and
	fine-tunes for embedding quality using contrastive learning.

	Self-guided mode (guide=None): the model's own embeddings are used as guide
	scores for false-negative filtering. Query prompts are naturally included
	in the guide computation since the same forward pass is reused.

	The tokenizer auto-appends EOS via TemplateProcessing.
	Query prompt is prepended only to anchor (query) texts via the `prompts` feature.

	Args:
		model_name_or_path: Path to Stage I checkpoint.
		guide_model_name_or_path: Guide model for false-negative filtering.
			If None, self-guided mode (model guides itself).
		query_prompt: Instruction prefix for queries (anchor column only).
		mini_batch_size: GPU memory batch size (actual forward pass size).
		per_device_train_batch_size: Effective batch size per GPU (via gradient caching).
		margin_strategy: False-negative filtering strategy ("absolute" or "relative").
		margin: Margin threshold. Negative values (e.g., -0.1) for self-guided mode.
	"""
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	setproctitle("Anchor-Embedding Stage2 SFT")

	if use_wandb:
		os.environ["WANDB_PROJECT"] = wandb_project

	if run_name is None:
		model_short = model_name_or_path.rstrip("/").split("/")[-1]
		run_name = (
			f"stage2_{model_short}"
			f"_lr{learning_rate}_bs{per_device_train_batch_size}"
			f"_mbs{mini_batch_size}_ep{num_epochs}"
		)

	if local_rank == 0:
		logger.info(
			f"=== Stage II SFT Training ===\n"
			f"  model:           {model_name_or_path}\n"
			f"  guide:           {guide_model_name_or_path or 'self-guided'}\n"
			f"  output_dir:      {output_dir}\n"
			f"  train_data:      {train_data}\n"
			f"  eval_data:       {eval_data}\n"
			f"  query_prompt:    {query_prompt!r}\n"
			f"  batch_size:      {per_device_train_batch_size}\n"
			f"  mini_batch_size: {mini_batch_size}\n"
			f"  learning_rate:   {learning_rate}\n"
			f"  num_epochs:      {num_epochs}\n"
			f"  max_seq_length:  {max_seq_length}\n"
			f"  margin_strategy: {margin_strategy}\n"
			f"  margin:          {margin}\n"
			f"  temperature:     {temperature}\n"
			f"  hardness_alpha:  {hardness_alpha}\n"
			f"  hardness_mode:   {hardness_mode}\n"
			f"  run_name:        {run_name}\n"
		)

	output_dir = f"{output_dir}/{run_name}"
	os.makedirs(output_dir, exist_ok=True)

	# --- Load training dataset ---
	logger.info(f"Loading training data from {train_data}...")
	if train_data.endswith(".jsonl"):
		train_dataset = hf_load_dataset("json", data_files=train_data, split="train")
	else:
		train_dataset = load_from_disk(train_data)

	if local_rank == 0:
		logger.info(f"Train dataset: {len(train_dataset)} samples")
		logger.info(f"  anchor:   {train_dataset[0]['anchor'][:120]}...")
		logger.info(f"  positive: {train_dataset[0]['positive'][:120]}...")

	# --- Load evaluation ---
	logger.info(f"Loading evaluation data from {eval_data}...")
	evaluator = load_eval_data(eval_data, query_prompt=query_prompt)

	# --- Build model (left padding + tokenizer auto-EOS + lasttoken pooling) ---
	logger.info(f"Building SentenceTransformer from {model_name_or_path}...")
	model = build_model(
		model_name_or_path,
		max_seq_length=max_seq_length,
		attn_implementation=attn_implementation,
	)

	if local_rank == 0:
		logger.info(f"Model architecture: {model}")
		logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

		# Verify tokenizer auto-EOS
		tokenizer = model.tokenizer
		test_ids = tokenizer.encode("hello")
		logger.info(
			f"Tokenizer EOS check: encode('hello') -> ...{test_ids[-3:]} "
			f"(last token should be EOS={tokenizer.eos_token_id})"
		)

	# --- Load guide model (if provided, otherwise self-guided) ---
	if guide_model_name_or_path is not None:
		logger.info(f"Loading guide model from {guide_model_name_or_path}...")
		guide = SentenceTransformer(
			guide_model_name_or_path,
			model_kwargs={"dtype": torch.bfloat16},
		)
		guide.max_seq_length = max_seq_length
	else:
		guide = None
		logger.info("Using self-guided mode (model guides itself)")

	# --- Loss function ---
	loss = CachedGISTEmbedLoss(
		model=model,
		guide=guide,
		mini_batch_size=mini_batch_size,
		temperature=temperature,
		margin_strategy=margin_strategy,
		margin=margin,
		contrast_anchors=False,
		contrast_positives=False,
		hardness_alpha=hardness_alpha,
		hardness_mode=hardness_mode,
	)

	# --- Training arguments ---
	# The `prompts` dict maps dataset column names to prompt prefixes.
	# Only anchor (query) gets the instruction prompt; positive/negative columns get none.
	args = SentenceTransformerTrainingArguments(
		output_dir=output_dir,
		num_train_epochs=num_epochs,
		per_device_train_batch_size=per_device_train_batch_size,
		per_device_eval_batch_size=per_device_eval_batch_size,
		gradient_accumulation_steps=gradient_accumulation_steps,
		learning_rate=learning_rate,
		warmup_ratio=warmup_ratio,
		logging_steps=logging_steps,
		save_strategy=save_strategy,
		save_steps=save_steps,
		eval_strategy=eval_strategy,
		eval_steps=eval_steps,
		fp16=not bf16,
		bf16=bf16,
		seed=seed,
		gradient_checkpointing=gradient_checkpointing,
		eval_on_start=eval_on_start,
		report_to="wandb" if use_wandb else [],
		run_name=run_name,
		dataloader_num_workers=dataloader_num_workers,
		prompts={"anchor": query_prompt},
	)

	# --- Trainer ---
	trainer = SentenceTransformerTrainer(
		model=model,
		args=args,
		train_dataset=train_dataset,
		loss=loss,
		evaluator=evaluator,
	)

	# --- Train ---
	trainer.train(resume_from_checkpoint=resume_from_checkpoint)
	trainer.evaluate()
	# --- Save final model ---
	trainer.save_model(f"{output_dir}/final")
	logger.info(f"Final model saved to {output_dir}/final")


if __name__ == "__main__":
	fire.Fire(train)
