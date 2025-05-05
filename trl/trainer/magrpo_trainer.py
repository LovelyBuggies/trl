# Copyright 2020-2025 Shuo Liu

import torch
from typing import Any, Callable, List, Optional, Union
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    Trainer,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
)
from .utils import selective_log_softmax
from ..extras.profiling import profiling_decorator
from ..models import unwrap_model_for_generation
from trl import GRPOTrainer, MAGRPOConfig

from contextlib import nullcontext

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[List[str], List[str]], List[float]]]

def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class MAGRPOTrainer(Trainer):
    """
    Multi-Agent Group Relative Policy Optimization Trainer (MAGRPO).
    Currently, only supports homogenous agents and shared reward functions.

    Args:
        model (str or PreTrainedModel): The model to be trained for homogenous agents
        num_agents (int): The number of agents.
        reward_funcs (RewardFunc or list[RewardFunc]): The reward functions for all agents.
        args (MAGRPOConfig, optional): The training arguments. If not provided, default arguments will be used.
        train_dataset (Dataset or IterableDataset, optional): The training dataset. If not provided, the default
            dataset will be used.
    """

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            num_agents: int,
            reward_funcs: Union[RewardFunc, List[RewardFunc]],
            args: Optional[MAGRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
    ):

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
        )

        self.num_agents = num_agents
        self.reward_funcs = reward_funcs
        self.agents = [model for _ in range(num_agents)]  # For homogenous agents

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            pad_token_id=processing_class.pad_token_id,
            bos_token_id=processing_class.bos_token_id,
            eos_token_id=processing_class.eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            cache_implementation=args.cache_implementation,
        )
        self.mask_truncated_completions = args.mask_truncated_completions

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Train the multi-agent model.

        This method coordinates the training of multiple agents, collecting
        outputs from each agent and computing rewards based on their interactions.
        """
        # Create the data pipeline for generating examples
        for epoch in range(int(self.args.num_train_epochs)):
            for batch_idx, batch in enumerate(self.get_train_dataloader()):
                # Get the prompts from the batch
                prompts = [sample["prompt"] for sample in batch]

                # Generate completions from each agent
                all_completions = []
                for agent_idx in range(self.num_agents):
                    agent_completions = self._generate_completions(
                        self.agents[agent_idx],
                        prompts,
                        num_return_sequences=self.args.num_generations,
                        max_length=self.args.max_completion_length,
                    )
                    all_completions.append(agent_completions)

                # Compute rewards based on all agents' completions
                rewards = self._compute_rewards(prompts, [agent_completions["completions"] for agent_completions in all_completions])

                # Update each agent using the rewards
                for agent_idx in range(self.num_agents):
                    self._compute_loss(self.agents[agent_idx], all_completions[agent_idx])

                # Log progress
                if batch_idx % self.args.logging_steps == 0:
                    self._log_training_progress(epoch, batch_idx, rewards)

                # Save checkpoints
                if batch_idx % self.args.save_steps == 0:
                    self._save_checkpoints(epoch, batch_idx)

    def _generate_completions(self, agent, prompts, num_return_sequences=1, max_length=128):
        """Generate completions from EACH agent given prompt."""
        prompt_inputs = self.processing_class(
            text=prompts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Generate completions using the agent
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (nullcontext()):
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config, num_return_sequences=num_return_sequences
                )

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        device = self.accelerator.device
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        return {
            "prompts": prompts,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completions": completions,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }

    def _compute_rewards(self, prompts, completions_list):
        if self.num_agents == 2 and callable(self.reward_funcs):
            return self.reward_funcs(completions_list[0], completions_list[1])
        else:
            raise NotImplementedError(
                "Currently, only the length ratio reward function for 2 agents is implemented."
            )

    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i: i + batch_size]
            attention_mask_batch = attention_mask[i: i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _compute_loss(self, agent, completions):
        """
        Compute the loss for EACH agent.
        """
        prompt_ids = completions["prompt_ids"]
        completion_ids = completions["completion_ids"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(agent, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def _log_training_progress(self, epoch, batch_idx, rewards):
        """Log training progress."""
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        self.log({
            "epoch": epoch,
            "batch": batch_idx,
            "average_reward": avg_reward,
        })

    def _save_checkpoints(self, epoch, batch_idx):
        """Save model checkpoints."""
        for agent_idx, agent in enumerate(self.agents):
            output_dir = f"{self.args.output_dir}/agent_{agent_idx}/epoch_{epoch}_batch_{batch_idx}"
            agent.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

    def save_model(self, output_dir):
        """Save the final trained models."""
        for agent_idx, agent in enumerate(self.agents):
            agent_dir = f"{output_dir}/agent_{agent_idx}"
            agent.save_pretrained(agent_dir)
            self.tokenizer.save_pretrained(agent_dir)

        print(f"All agent models saved to {output_dir}")
