import logging
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
    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
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
            pooling_mode=model_args.pooling_mode,
            train_batch_size=training_args.per_device_train_batch_size
            * accelerator.num_processes
            * training_args.gradient_accumulation_steps,
            max_seq_length=model_args.max_seq_length,
            bidirectional=model_args.bidirectional,
            epochs=training_args.num_train_epochs,
            seed=training_args.seed,
            warmup_steps=training_args.warmup_steps,
            lr=training_args.learning_rate,
            lora_r=custom_args.lora_r,
            use_peft=model_args.use_peft,
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

    trainer.train()


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
