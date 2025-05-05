# Copyright 2020-2025 Shuo Liu

from typing import Any, Callable, Optional, Union
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    is_wandb_available,
)
from transformers.utils import is_peft_available
from .grpo_trainer import GRPOTrainer
from .magrpo_config import MAGRPOConfig


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class MAGRPOTrainer(GRPOTrainer):
    """
    Trainer for the Multi-Agent Group Relative Policy Optimization (GRPO) method.
    """

    # a same as GRPOTrainer
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[MAGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
        )