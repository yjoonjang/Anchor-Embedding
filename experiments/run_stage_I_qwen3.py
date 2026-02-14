import logging
import math
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.nn import CrossEntropyLoss

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from accelerate import Accelerator
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model

from llm2vec import LLM2Vec_q2d_d2q
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.utils import load_loss
from llm2vec.experiment_utils import generate_experiment_id

from tqdm import tqdm

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def prepare_for_tokenization_qwen3(model, text, pooling_mode="eos_token"):
    text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    return text


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The base model checkpoint for weights initialization."},
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The PEFT model checkpoint to add on top of base model."},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable bidirectional attention."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default torch.dtype and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": "The attention implementation to use.",
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="eos_token",
        metadata={
            "help": "The pooling mode to use.",
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use PEFT or not."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5, KoEnQP"},
    )
    dataset_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file or folder."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples for debugging."},
    )


@dataclass
class CustomArguments:
    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )
    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})
    stop_after_n_steps: Optional[int] = field(
        default=None, metadata={"help": "Stop training after n steps (None to disable)"}
    )
    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )
    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={"help": "The loss class to use for training."},
    )
    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )
    d2q_weight: float = field(
        default=0.8, metadata={"help": "Weight for D2Q (document-to-query) reconstruction loss"}
    )
    q2d_weight: float = field(
        default=0.2, metadata={"help": "Weight for Q2D (query-to-document) reconstruction loss"}
    )
    retrieval_eval_steps: float = field(
        default=0, metadata={"help": "Run NanoBEIR evaluation every N steps. If <1, interpreted as ratio of total steps (0 to disable)"}
    )
    retrieval_eval_on_start: bool = field(
        default=False, metadata={"help": "Run NanoBEIR evaluation before training starts"}
    )
    retrieval_eval_batch_size: int = field(
        default=64, metadata={"help": "Batch size for NanoBEIR evaluation encoding"}
    )


@dataclass
class DefaultCollator:
    model: LLM2Vec_q2d_d2q

    def __init__(self, model: LLM2Vec_q2d_d2q) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                text = prepare_for_tokenization_qwen3(
                    self.model, text, pooling_mode=self.model.pooling_mode
                )
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class NanoBEIREvalCallback(TrainerCallback):
    """Evaluates retrieval performance on NanoBEIR (ko+en, MSMARCO+NQ) during training."""

    DATASET_CONFIGS = {
        "ko": {
            "repo": "sionic-ai/NanoBEIR-ko",
            "subsets": ["NanoMSMARCO", "NanoNQ"],
            "instruction": "주어진 질문에 대해 관련 문서를 검색하세요",
        },
        "en": {
            "repo": "sentence-transformers/NanoBEIR-en",
            "subsets": ["NanoMSMARCO", "NanoNQ"],
            "instruction": "Retrieve relevant documents for the given query",
        },
    }

    def __init__(self, model, eval_steps: float, eval_batch_size: int = 64, eval_on_start: bool = False):
        self.model = model
        self._eval_steps_raw = eval_steps  # raw value, might be ratio (<1)
        self.eval_steps = None  # resolved to int in on_train_begin
        self.eval_batch_size = eval_batch_size
        self.eval_on_start = eval_on_start
        self.datasets = {}
        self._loaded = False

    def _run_eval(self, args, state):
        """Run evaluation and return metrics dict."""
        if args.local_process_index != 0:
            return

        if not self._loaded:
            self._load_datasets()

        logger.info(f"Running NanoBEIR evaluation at step {state.global_step}")

        was_training = self.model.training
        self.model.eval()

        all_metrics = {}

        for key, data in self.datasets.items():
            lang, subset = key.split("/")
            instruction = data["instruction"]
            queries = data["queries"]
            corpus = data["corpus"]
            qrels = data["qrels"]

            query_texts = [
                f"{instruction}; !@#$%^&*(){q['text']}" for q in queries
            ]
            corpus_texts = [f"!@#$%^&*(){c['text']}" for c in corpus]

            query_ids = [q["_id"] for q in queries]
            corpus_ids = [c["_id"] for c in corpus]

            query_embs = self._encode(query_texts, self.eval_batch_size)
            corpus_embs = self._encode(corpus_texts, self.eval_batch_size)

            metrics = self._compute_metrics(
                query_embs, corpus_embs, qrels, query_ids, corpus_ids
            )

            for metric_name, value in metrics.items():
                all_metrics[f"eval/{lang}/{subset}/{metric_name}"] = value

            logger.info(
                f"  NanoBEIR {key}: NDCG@10={metrics['ndcg@10']:.4f}, MRR@10={metrics['mrr@10']:.4f}"
            )

        # Compute averages across all datasets
        ndcg_vals = [v for k, v in all_metrics.items() if "ndcg@10" in k]
        mrr_vals = [v for k, v in all_metrics.items() if "mrr@10" in k]
        if ndcg_vals:
            all_metrics["eval/avg/ndcg@10"] = sum(ndcg_vals) / len(ndcg_vals)
        if mrr_vals:
            all_metrics["eval/avg/mrr@10"] = sum(mrr_vals) / len(mrr_vals)

        # Log to WandB
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(all_metrics, step=state.global_step)
        except ImportError:
            pass

        if was_training:
            self.model.train()

    def on_train_begin(self, args, state, control, **kwargs):
        # Resolve ratio-based eval_steps to absolute step count
        if self._eval_steps_raw > 0 and self._eval_steps_raw < 1:
            self.eval_steps = max(1, int(state.max_steps * self._eval_steps_raw))
            logger.info(
                f"NanoBEIR eval_steps resolved: {self._eval_steps_raw} * {state.max_steps} = {self.eval_steps} steps"
            )
        else:
            self.eval_steps = int(self._eval_steps_raw)

        if self.eval_on_start:
            self._run_eval(args, state)

    def _load_datasets(self):
        from datasets import load_dataset as hf_load_dataset

        for lang, cfg in self.DATASET_CONFIGS.items():
            for subset in cfg["subsets"]:
                key = f"{lang}/{subset}"
                try:
                    queries = hf_load_dataset(cfg["repo"], "queries", split=subset)
                    corpus = hf_load_dataset(cfg["repo"], "corpus", split=subset)
                    qrels = hf_load_dataset(cfg["repo"], "qrels", split=subset)
                    self.datasets[key] = {
                        "queries": queries,
                        "corpus": corpus,
                        "qrels": qrels,
                        "instruction": cfg["instruction"],
                    }
                    logger.info(
                        f"Loaded NanoBEIR {key}: {len(queries)} queries, {len(corpus)} corpus docs"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load NanoBEIR {key}: {e}")
        self._loaded = True

    @torch.no_grad()
    def _encode(self, texts, batch_size):
        """Encode texts using the model with eos_token pooling."""
        all_embeddings = []
        device = next(self.model.parameters()).device

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            prepared = [
                prepare_for_tokenization_qwen3(self.model, t) for t in batch_texts
            ]
            features = self.model.tokenize(prepared)
            features = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }
            embeddings = self.model(features)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _compute_metrics(self, query_embs, corpus_embs, qrels, query_ids, corpus_ids, k=10):
        """Compute NDCG@k and MRR@k."""
        query_embs = torch.nn.functional.normalize(query_embs, p=2, dim=1)
        corpus_embs = torch.nn.functional.normalize(corpus_embs, p=2, dim=1)

        sim = torch.mm(query_embs, corpus_embs.t())

        # Build relevance mapping: query_id -> {corpus_id: score}
        relevance = {}
        for item in qrels:
            qid = str(item["query-id"])
            cid = str(item["corpus-id"])
            score = item.get("score", 1)
            if qid not in relevance:
                relevance[qid] = {}
            relevance[qid][cid] = score

        ndcg_scores = []
        mrr_scores = []

        for q_idx, qid in enumerate(query_ids):
            qid_str = str(qid)
            if qid_str not in relevance:
                continue

            top_k = torch.topk(sim[q_idx], min(k, sim.shape[1]))
            top_k_indices = top_k.indices.tolist()

            rel_labels = []
            for idx in top_k_indices:
                cid = str(corpus_ids[idx])
                rel_labels.append(relevance[qid_str].get(cid, 0))

            # MRR@k
            mrr = 0.0
            for rank, rel in enumerate(rel_labels, 1):
                if rel > 0:
                    mrr = 1.0 / rank
                    break
            mrr_scores.append(mrr)

            # NDCG@k
            dcg = sum(
                rel / math.log2(rank + 1)
                for rank, rel in enumerate(rel_labels, 1)
            )
            ideal_rels = sorted(relevance[qid_str].values(), reverse=True)[:k]
            idcg = sum(
                rel / math.log2(rank + 1)
                for rank, rel in enumerate(ideal_rels, 1)
            )
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return {
            "ndcg@10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
            "mrr@10": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        }

    def on_step_end(self, args, state, control, **kwargs):
        if not self.eval_steps or self.eval_steps <= 0:
            return
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return
        self._run_eval(args, state)

    def on_train_end(self, args, state, control, **kwargs):
        self._run_eval(args, state)


class LLM2VecSupervisedTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_function=None,
        d2q_weight: float = 0.8,
        q2d_weight: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.d2q_weight = d2q_weight
        self.q2d_weight = q2d_weight
        self._custom_logs = {}

    def log(self, logs, *args, **kwargs):
        if self._custom_logs:
            logs.update(self._custom_logs)
            self._custom_logs = {}
        super().log(logs, *args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs
        unwrapped = _unwrap_model(model)

        q_reps = model(features[0])
        d_reps = model(features[1])

        q_input_ids = features[0]["input_ids"]
        q_attention_mask = features[0]["attention_mask"]
        d_input_ids = features[1]["input_ids"]
        d_attention_mask = features[1]["attention_mask"]

        q_labels = q_input_ids.clone()
        q_labels[q_labels == unwrapped.tokenizer.eos_token_id] = -100

        d_labels = d_input_ids.clone()
        d_labels[d_labels == unwrapped.tokenizer.eos_token_id] = -100

        attention_mask_tmp = torch.full(
            (d_attention_mask.shape[0], 1),
            1,
            dtype=d_attention_mask.dtype,
            device=d_attention_mask.device,
        )
        d2q_attention_mask = torch.cat(
            (attention_mask_tmp, q_attention_mask), dim=1
        )
        q2d_attention_mask = torch.cat(
            (attention_mask_tmp, d_attention_mask), dim=1
        )

        label_tmp = torch.full(
            (d_labels.shape[0], 1),
            -100,
            dtype=d_labels.dtype,
            device=d_labels.device,
        )
        d2q_labels = torch.cat((label_tmp, q_labels), dim=1)
        q2d_labels = torch.cat((label_tmp, d_labels), dim=1)

        def compute_loss_q2d_d2q(logits, labels):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, unwrapped.model.config.vocab_size
            )
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            return loss_fct(shift_logits, shift_labels)

        # Ensure consistent dtype for the reconstruction forward passes
        model_dtype = unwrapped.model.embed_tokens.weight.dtype

        # D2Q: use document embedding to reconstruct query via teacher forcing
        q_input_embeds = unwrapped.model.embed_tokens(q_input_ids)
        combined_d2q_input_embeds = torch.cat(
            [d_reps.unsqueeze(1).to(model_dtype), q_input_embeds], dim=1
        )
        d2q_outputs = unwrapped.model(
            inputs_embeds=combined_d2q_input_embeds,
            attention_mask=d2q_attention_mask,
        )
        d2q_logits = unwrapped.lm_head(d2q_outputs[0].to(unwrapped.lm_head.weight.dtype))
        d2q_loss = compute_loss_q2d_d2q(d2q_logits, d2q_labels)

        # Q2D: use query embedding to reconstruct document via teacher forcing
        d_input_embeds = unwrapped.model.embed_tokens(d_input_ids)
        combined_q2d_input_embeds = torch.cat(
            [q_reps.unsqueeze(1).to(model_dtype), d_input_embeds], dim=1
        )
        q2d_outputs = unwrapped.model(
            inputs_embeds=combined_q2d_input_embeds,
            attention_mask=q2d_attention_mask,
        )
        q2d_logits = unwrapped.lm_head(q2d_outputs[0].to(unwrapped.lm_head.weight.dtype))
        q2d_loss = compute_loss_q2d_d2q(q2d_logits, q2d_labels)

        loss = self.d2q_weight * d2q_loss + self.q2d_weight * q2d_loss

        self._custom_logs = {
            "d2q_loss": d2q_loss.detach().item(),
            "q2d_loss": q2d_loss.detach().item(),
        }

        return loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = lambda worker_id: seed_worker(worker_id, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        unwrapped = _unwrap_model(self.model)
        unwrapped.save(output_dir)

        # Also save as standard state_dict for HF Trainer resume compatibility
        model_state = unwrapped.model.state_dict()
        torch.save(model_state, os.path.join(output_dir, "pytorch_model.bin"))

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if custom_args.experiment_id is not None:
        experiment_id = custom_args.experiment_id
    else:
        experiment_id = generate_experiment_id(
            name=data_args.dataset_name,
            split="train",
            model_name=(
                model_args.model_name_or_path
                if "/" not in model_args.model_name_or_path
                else model_args.model_name_or_path.split("/")[-1]
            ),
            train_batch_size=training_args.per_device_train_batch_size,
            max_seq_length=model_args.max_seq_length,
            epochs=training_args.num_train_epochs,
            warmup_steps=training_args.warmup_steps,
            lr=training_args.learning_rate,
            lora_r=custom_args.lora_r,
            use_peft=model_args.use_peft,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            d2q_weight=custom_args.d2q_weight,
            q2d_weight=custom_args.q2d_weight,
            num_gpus=accelerator.num_processes,
        )

    training_args.output_dir = f"{training_args.output_dir}/{experiment_id}"

    train_dataset = load_dataset(
        data_args.dataset_name,
        split="train",
        file_path=data_args.dataset_file_path,
        effective_batch_size=training_args.per_device_train_batch_size
        * accelerator.num_processes,
    )

    train_examples = [
        train_dataset[i]
        for i in tqdm(
            range(len(train_dataset)),
            desc="Loading train examples...",
            disable=not accelerator.is_main_process,
        )
    ]

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = LLM2Vec_q2d_d2q.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    # Qwen3 tokenizer already has pad_token set, but we override to match
    # the eos_token convention used by the codebase
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    if model_args.use_peft:
        model.model = initialize_peft(
            model.model,
            lora_r=custom_args.lora_r,
            lora_alpha=2 * custom_args.lora_r,
            lora_dropout=custom_args.lora_dropout,
        )
    else:
        print("Not using PEFT - full finetuning")

    tokenizer = model.tokenizer

    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    data_collator = DefaultCollator(model)

    trainer = LLM2VecSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=train_loss,
        d2q_weight=custom_args.d2q_weight,
        q2d_weight=custom_args.q2d_weight,
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    if custom_args.retrieval_eval_steps > 0:
        trainer.add_callback(
            NanoBEIREvalCallback(
                model=model,
                eval_steps=custom_args.retrieval_eval_steps,
                eval_batch_size=custom_args.retrieval_eval_batch_size,
                eval_on_start=custom_args.retrieval_eval_on_start,
            )
        )

    # Auto-detect checkpoint for resume
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)


def initialize_peft(
    model,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print("Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


if __name__ == "__main__":
    main()
