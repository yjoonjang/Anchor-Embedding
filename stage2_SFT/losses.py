from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states
from transformers import PreTrainedTokenizerBase

from sentence_transformers.models import StaticEmbedding
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedGISTEmbedLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                if reps_mb.requires_grad:
                    surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                    surrogate.backward()


class CachedGISTEmbedLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer | None = None,
        temperature: float = 0.01,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
        margin_strategy: Literal["absolute", "relative"] = "absolute",
        margin: float = 0.0,
        contrast_anchors: bool = True,
        contrast_positives: bool = True,
        gather_across_devices: bool = False,
    ) -> None:
        """
        CachedGISTEmbedLoss with self-guided mode support.

        When guide=None, the model itself is used as the guide (self-guided mode),
        reusing student embeddings as guide embeddings without a second forward pass.

        Args:
            model: SentenceTransformer model
            guide: Guide model for in-batch negative selection. If None, self-guided mode.
            temperature: Temperature for cosine similarity scaling.
            mini_batch_size: Mini-batch size for forward pass (memory control).
            margin_strategy: "absolute" or "relative" for false-negative filtering.
            margin: Margin threshold. For self-guided, negative values (e.g., -0.1) recommended.
            contrast_anchors: Include anchor-anchor pairs in loss.
            contrast_positives: Include positive-positive pairs in loss.
            gather_across_devices: Gather embeddings across GPUs for larger effective batch.
        """
        super().__init__()
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "CachedGISTEmbedLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding. "
                "Consider using GISTEmbedLoss instead."
            )
        self.model = model
        self.temperature = temperature
        self.similarity_fct = nn.CosineSimilarity(dim=-1)

        if guide is None:
            self.guide = model
            self.is_self_guided = True
        else:
            self.guide = guide
            self.is_self_guided = model is guide

        if self.is_self_guided:
            if not hasattr(model, "tokenizer"):
                raise ValueError("The training model must have a tokenizer attribute.")
            if not isinstance(model.tokenizer, PreTrainedTokenizerBase):
                raise ValueError("The training model must use a PreTrainedTokenizer from transformers.")
        else:
            if not hasattr(model, "tokenizer") or not hasattr(self.guide, "tokenizer"):
                raise ValueError("Both the training model and the guiding model must have a tokenizer attribute.")
            if not isinstance(model.tokenizer, PreTrainedTokenizerBase) or not isinstance(
                self.guide.tokenizer, PreTrainedTokenizerBase
            ):
                raise ValueError(
                    "Both the training model and the guiding model must use a PreTrainedTokenizer from transformers."
                )

        self.mini_batch_size = mini_batch_size
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
        self.show_progress_bar = show_progress_bar

        if self.is_self_guided:
            self.must_retokenize = False
        else:
            self.must_retokenize = (
                model.tokenizer.vocab != guide.tokenizer.vocab or guide.max_seq_length < model.max_seq_length
            )
        if self.must_retokenize:
            self.tokenizer = model.tokenizer
        if margin_strategy not in ("absolute", "relative"):
            raise ValueError("margin_strategy must be 'absolute' or 'relative'.")
        self.margin_strategy = margin_strategy
        self.margin = margin
        self.contrast_anchors = contrast_anchors
        self.contrast_positives = contrast_positives
        self.gather_across_devices = gather_across_devices
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def sim_matrix(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        return self.similarity_fct(embed1.unsqueeze(1), embed2.unsqueeze(0))

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, Tensor, RandContext | None]:
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]

            if self.is_self_guided:
                guide_reps = reps.detach()
            else:
                with torch.no_grad():
                    if self.must_retokenize:
                        decoded = self.tokenizer.batch_decode(
                            sentence_feature_minibatch["input_ids"], skip_special_tokens=True
                        )
                        sentence_feature_minibatch = self.guide.tokenize(decoded)
                        sentence_feature_minibatch = {
                            key: value.to(self.guide.device) for key, value in sentence_feature_minibatch.items()
                        }
                    guide_reps = self.guide(sentence_feature_minibatch)["sentence_embedding"]

        return reps, guide_reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, Tensor, RandContext | None]]:
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, begin in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            end = begin + self.mini_batch_size
            reps, guide_reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, guide_reps, random_state

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]]) -> Tensor:
        loss = self.calculate_loss(reps, reps_guided, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def calculate_loss(
        self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]], with_backward: bool = False
    ) -> Tensor:
        if len(reps) != len(reps_guided):
            raise ValueError("reps and reps_guided must have the same length")

        anchors = torch.cat(reps[0])
        anchors_guide = torch.cat(reps_guided[0])
        candidates = [torch.cat(r) for r in reps[1:]]
        candidates_guide = [torch.cat(r) for r in reps_guided[1:]]

        batch_size = anchors.size(0)
        offset = 0

        if self.gather_across_devices:
            candidates = [all_gather_with_grad(candidate) for candidate in candidates]
            candidates_guide = [all_gather_with_grad(candidate) for candidate in candidates_guide]

            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        range_labels = torch.arange(offset, offset + batch_size, device=anchors.device)

        losses: list[torch.Tensor] = []
        for begin in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            end = begin + self.mini_batch_size

            ap_sim = self.sim_matrix(anchors[begin:end], candidates[0])
            guided_ap_sim = self.sim_matrix(anchors_guide[begin:end], candidates_guide[0])

            guided_sim = guided_ap_sim.diagonal(offset=offset + begin).view(-1, 1)

            def mask_false_negatives(guided_sim_mat, sim_mat, positive_mask: Tensor | None = None):
                if self.margin_strategy == "absolute":
                    mask = guided_sim_mat > (guided_sim - self.margin)
                elif self.margin_strategy == "relative":
                    mask = guided_sim_mat > (guided_sim * (1 - self.margin))

                if positive_mask is not None:
                    mask = mask & ~positive_mask
                sim_mat[mask] = -torch.inf
                return sim_mat

            positive_mask = torch.eye(*guided_ap_sim.shape, dtype=torch.bool, device=guided_ap_sim.device)
            positive_mask = positive_mask.roll(begin)

            ap_sim = mask_false_negatives(guided_ap_sim, ap_sim, positive_mask=positive_mask)
            scores = [ap_sim]

            if self.contrast_anchors:
                aa_sim = self.sim_matrix(anchors[begin:end], anchors)
                guided_aa_sim = self.sim_matrix(anchors_guide[begin:end], anchors_guide)
                aa_sim = mask_false_negatives(guided_aa_sim, aa_sim)
                scores.append(aa_sim)

            if self.contrast_positives:
                pp_sim = self.sim_matrix(
                    candidates[0][offset + begin : min(offset + end, offset + batch_size)], candidates[0]
                )
                guided_pp_sim = self.sim_matrix(
                    candidates_guide[0][offset + begin : min(offset + end, offset + batch_size)], candidates_guide[0]
                )
                pp_sim = mask_false_negatives(guided_pp_sim, pp_sim)
                scores.append(pp_sim)

            if len(candidates) > 1:
                for i in range(1, len(candidates)):
                    neg_sim = self.sim_matrix(anchors[begin:end], candidates[i])
                    guided_neg_sim = self.sim_matrix(anchors_guide[begin:end], candidates_guide[i])
                    neg_sim = mask_false_negatives(guided_neg_sim, neg_sim)
                    scores.append(neg_sim)

            scores = torch.cat(scores, dim=1)
            scores = scores / self.temperature
            loss_mbatch: torch.Tensor = (
                self.cross_entropy_loss(scores, range_labels[begin:end]) * len(scores) / batch_size
            )
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        loss = sum(losses)
        return loss

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        reps = []
        reps_guided = []
        self.random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            reps_guided_mbs = []
            random_state_mbs = []
            for reps_mb, reps_guided_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                reps_guided_mbs.append(reps_guided_mb.detach())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            reps_guided.append(reps_guided_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps, reps_guided)
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            loss = self.calculate_loss(reps, reps_guided)
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "guide": self.guide,
            "temperature": self.temperature,
            "mini_batch_size": self.mini_batch_size,
            "margin_strategy": self.margin_strategy,
            "margin": self.margin,
            "contrast_anchors": self.contrast_anchors,
            "contrast_positives": self.contrast_positives,
            "gather_across_devices": self.gather_across_devices,
        }
