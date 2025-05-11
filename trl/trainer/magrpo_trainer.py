import torch
from typing import Callable, List, Optional, Union, Dict, Any
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from torch.utils.data import DataLoader
from trl import MAGRPOConfig
import numpy as np
import logging
import os
import wandb  # Add wandb import

RewardFunc = Union[str, PreTrainedModel, Callable[[List[str], List[str]], List[float]]]


STOPWORDS = set([
        "a", "an", "the", "and", "but", "or", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "when", "who", "how", "where", "why", "is", "am", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "to", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "from", "up",
        "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "can", "will", "should", "now", "of"
    ])

class MAGRPOTrainer:
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
        wandb_config (dict, optional): Configuration for Weights & Biases logging.
    """

    def __init__(
            self,
            model: Optional[Union[str, PreTrainedModel]] = None,
            agents: Optional[List[PreTrainedModel]] = None,
            num_agents: int = 2,
            reward_funcs: Union[RewardFunc, List[RewardFunc]] = None,
            args: Optional[MAGRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            wandb_config: Optional[Dict[str, Any]] = None,
    ):
        # Validate inputs
        if model is None and agents is None:
            raise ValueError("Either model or agents must be provided")
        if model is not None and agents is not None:
            raise ValueError("Cannot provide both model and agents parameters")

        # Set up reward functions and args
        self.reward_funcs = reward_funcs
        self.args = args if args is not None else MAGRPOConfig()

        # Initialize agents
        if agents is not None:
            # Use pre-created agents
            self.agents = agents
            self.num_agents = len(agents)
            # Set model_name based on the first agent
            if hasattr(agents[0], 'base_model') and hasattr(agents[0].base_model, 'config') and hasattr(
                    agents[0].base_model.config, 'model_type'):
                # For PEFT models
                self.model_name = agents[0].base_model.config.model_type
            else:
                # For regular models
                self.model_name = agents[0].__class__.__name__
        else:
            # Create agents from model
            self.num_agents = num_agents

            if isinstance(model, str):
                # Create agents from model name
                from transformers import AutoModelForCausalLM
                self.agents = [AutoModelForCausalLM.from_pretrained(model) for _ in range(num_agents)]
                self.model_name = model
            else:
                # Create TRUE deep copies of the model for each agent
                import copy
                self.agents = []
                self.model_name = model.__class__.__name__
                for _ in range(num_agents):
                    # Create a new instance with the same configuration
                    agent_copy = type(model)(model.config)
                    # Copy the state dictionary to transfer parameters
                    agent_copy.load_state_dict(copy.deepcopy(model.state_dict()))
                    self.agents.append(agent_copy)

        # Validate agents
        if self.num_agents < 2:
            raise ValueError("MAGRPO requires at least 2 agents for training.")
        if self.args.num_generations < 2:
            raise ValueError("MAGRPO requires num_generations to be at least 2 for multi-agent training.")

        # Set up dataset and tokenizer
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize optimizers for each agent
        self.optimizers = [torch.optim.AdamW(
            agent.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        ) for agent in self.agents]

        # Initialize wandb
        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if self.wandb_config is not None:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases for tracking."""
        if not self.wandb_initialized:
            # Set default wandb config if not provided
            if self.wandb_config is None:
                self.wandb_config = {}

            # Set up default values
            wandb_project = self.wandb_config.get("project", "trl")
            wandb_entity = self.wandb_config.get("entity", "nu-llpr")
            wandb_name = self.wandb_config.get("name", "test-magrpo")

            # Create a dictionary with all training configurations for wandb
            config_dict = {
                "model_name": self.model_name,
                "num_agents": self.num_agents,
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "num_train_epochs": self.args.num_train_epochs,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,  # Changed from max_completion_length
            }

            # Initialize wandb
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                config=config_dict
            )

            self.wandb_initialized = True
            self.logger.info(
                f"Initialized wandb with project={wandb_project}, entity={wandb_entity}, name={wandb_name}")

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Create a custom collate function that handles raw prompts
        def collate_fn(examples):
            # For datasets with only 'prompt' field, we don't need tokenization here
            # We'll just return the batch as-is and handle tokenization in _generate_completions
            if all(isinstance(ex, dict) and 'prompt' in ex for ex in examples):
                return examples
            # For more complex datasets that might already have tokenized fields
            else:
                if self.tokenizer:
                    return DataCollatorWithPadding(self.tokenizer)(examples)
                else:
                    return examples

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Train the multi-agent model.

        This method coordinates the training of multiple agents, collecting
        outputs from each agent and computing rewards based on their interactions.

        Args:
            resume_from_checkpoint (str, optional): Path to a checkpoint to resume training from.
            **kwargs: Additional arguments to pass to the model during generation.
        """
        # Initialize wandb if not already done
        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        # Setup devices for training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for agent in self.agents:
            agent.to(device)
            agent.train()

        # Load checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self._load_from_checkpoint(resume_from_checkpoint)

        # Create the data pipeline for generating examples
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            epoch_loss = 0.0
            epoch_rewards = []
            epoch_agent_rewards = [[] for _ in range(self.num_agents)]

            for batch_idx, batch in enumerate(self.get_train_dataloader()):
                # If batch is already a list of dictionaries, we work with individual prompts
                if isinstance(batch, list) and all(isinstance(item, dict) for item in batch):
                    # Process each prompt separately - this is the expected format
                    for prompt_idx, prompt_item in enumerate(batch):
                        prompt = prompt_item["prompt"]

                        # Generate completions from each agent for this single prompt
                        all_completions = []
                        for agent_idx in range(self.num_agents):
                            # Zero gradients for each agent
                            self.optimizers[agent_idx].zero_grad()

                            agent_completions = self._generate_completions(
                                self.agents[agent_idx],
                                [prompt],  # Pass as a single-item list
                                num_return_sequences=self.args.num_generations,
                                max_new_tokens=self.args.max_new_tokens,  # Changed from max_completion_length
                                **kwargs
                            )
                            all_completions.append(agent_completions)

                        # Extract completions for reward calculation
                        agent_completions_list = []
                        for agent_idx in range(self.num_agents):
                            agent_completions_list.append(all_completions[agent_idx]["completions"][0])

                        # Compute rewards based on all agents' completions for this prompt
                        rewards = self._compute_rewards([prompt], agent_completions_list)
                        epoch_rewards.extend(rewards)

                        # Track rewards per agent for more detailed logging
                        for agent_idx in range(self.num_agents):
                            for reward in rewards:
                                epoch_agent_rewards[agent_idx].append(reward)
                                epoch_agent_rewards[agent_idx].append(reward)

                        # Update each agent using the rewards with proper gradient tracking
                        batch_loss = 0.0
                        agent_losses = []
                        for agent_idx in range(self.num_agents):
                            # Compute loss with the new function that enables proper gradient tracking
                            agent_loss = self._compute_loss_with_gradients(
                                self.agents[agent_idx],
                                all_completions[agent_idx],
                                rewards  # Invert rewards for agent1
                            )

                            # Backward pass and optimization
                            agent_loss.backward()
                            self.optimizers[agent_idx].step()

                            batch_loss += agent_loss.detach().item()
                            agent_losses.append(agent_loss.detach().item())

                        epoch_loss += batch_loss

                        # Log to wandb per batch
                        if self.wandb_initialized:
                            wandb.log({
                                "batch_loss": batch_loss,
                                "agent1_loss": agent_losses[0] if len(agent_losses) > 0 else 0,
                                "agent2_loss": agent_losses[1] if len(agent_losses) > 1 else 0,
                                "batch_rewards_mean": np.mean(rewards) if rewards else 0,
                                "step": epoch * len(self.get_train_dataloader()) + batch_idx,
                            })

                            # Log a sample of completions periodically
                            if batch_idx % 10 == 0 and rewards:
                                for agent_idx in range(self.num_agents):
                                    sample_completions = agent_completions_list[agent_idx]
                                    if sample_completions:
                                        wandb.log({
                                            f"agent{agent_idx + 1}_completion_length": len(
                                                sample_completions[0]) if sample_completions else 0
                                        })

                # Log progress
                if batch_idx % self.args.logging_steps == 0:
                    self._log_training_progress(epoch, batch_idx, epoch_rewards[-len(batch):] if epoch_rewards else [])

                # Save checkpoints
                if batch_idx % self.args.save_steps == 0:
                    self._save_checkpoints(epoch, batch_idx)

            # Log epoch summary
            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            avg_agent_rewards = [sum(rewards) / len(rewards) if rewards else 0 for rewards in epoch_agent_rewards]

            epoch_log = {
                "epoch": epoch,
                "epoch_loss": epoch_loss / len(self.get_train_dataloader()) if epoch_loss else 0,
                "epoch_avg_reward": avg_reward,
            }

            # Add agent-specific reward tracking
            for i, avg_agent_reward in enumerate(avg_agent_rewards):
                epoch_log[f"agent{i + 1}_avg_reward"] = avg_agent_reward

            if self.wandb_initialized:
                wandb.log(epoch_log)

            self.logger.info(f"Epoch {epoch}: Loss={epoch_loss / len(self.get_train_dataloader())}, "
                             f"Avg Reward={avg_reward}, "
                             f"Agent1 Avg Reward={avg_agent_rewards[0]}, "
                             f"Agent2 Avg Reward={avg_agent_rewards[1]}")

        # Close wandb
        if self.wandb_initialized:
            wandb.finish()

    # 1. In the _generate_completions method, change max_length to max_new_tokens
    def _generate_completions(self, agent, prompts, num_return_sequences=1, max_new_tokens=128, **kwargs):
        """
        Generate completions from an agent given prompts, preserving model state.

        Args:
            agent (PreTrainedModel): The agent model to generate completions.
            prompts (List[str]): List of prompts to generate completions for.
            num_return_sequences (int, optional): Number of completions to generate per prompt.
            max_new_tokens (int, optional): Maximum number of new tokens to generate.
            **kwargs: Additional arguments to pass to the model during generation.

        Returns:
            Dict: A dictionary containing generated completions and associated data.
        """
        device = agent.device
        batch_size = len(prompts)

        # Ensure tokenizer exists
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        # Store original model state and gradient settings
        training_mode = agent.training
        original_requires_grad = {}

        # Save original requires_grad states
        for name, param in agent.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False  # Temporarily disable gradients for generation

        agent.eval()  # Set to eval mode for generation

        # Generate completions without gradients
        generation_output = None
        try:
            # Use max_new_tokens instead of max_length
            generation_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,  # Changed from max_length
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            # If requesting multiple sequences, use sampling for diversity
            if num_return_sequences > 1:
                generation_kwargs.update({
                    "do_sample": True,  # Enable sampling for randomness
                    "temperature": 0.7,  # Control randomness (higher = more random)
                    "top_p": 0.9,  # Nucleus sampling
                    "top_k": 50,  # Limit vocabulary to top k tokens
                    "num_beams": 1,  # Disable beam search when sampling
                    "num_return_sequences": num_return_sequences,
                })

            # Add any additional user-provided kwargs
            generation_kwargs.update(kwargs)

            generation_output = agent.generate(**generation_kwargs)
        except Exception as e:
            # Restore model state before raising exception
            agent.train(training_mode)
            # Restore original requires_grad states
            for name, param in agent.named_parameters():
                if name in original_requires_grad:
                    param.requires_grad = original_requires_grad[name]
            raise ValueError(f"Generation failed: {str(e)}")

        # Restore original model state and gradients
        agent.train(training_mode)
        for name, param in agent.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        # Extract completion tokens (excluding prompt tokens)
        completion_input_ids = generation_output.sequences

        # For each prompt, we need to find its actual length in tokens
        # to properly extract just the completion part
        prompt_lengths = []
        for b in range(batch_size):
            # Get the prompt length by finding where padding starts or using full length
            prompt_len = prompt_input_ids[b].shape[0]
            # Find where padding token starts if any
            pad_positions = (prompt_input_ids[b] == self.tokenizer.pad_token_id).nonzero()
            if pad_positions.shape[0] > 0:
                # Use the position of the first padding token
                prompt_len = pad_positions[0].item()
            prompt_lengths.append(prompt_len)

        # Extract completion text
        completions = []
        completion_tokens_list = []

        # Calculate total sequence count
        total_sequences = completion_input_ids.shape[0]

        # Ensure this matches expected count
        if total_sequences != batch_size * num_return_sequences:
            self.logger.warning(f"Expected {batch_size * num_return_sequences} sequences but got {total_sequences}")

        # Process each prompt and its multiple completions
        for b in range(batch_size):
            prompt_len = prompt_lengths[b]
            batch_completions = []
            batch_completion_tokens = []

            # Get all sequences for this prompt
            start_idx = b * num_return_sequences
            end_idx = start_idx + num_return_sequences

            # Ensure we don't go out of bounds
            end_idx = min(end_idx, total_sequences)

            for s in range(start_idx, end_idx):
                # Get only the completion part (exclude the prompt tokens)
                completion_tokens = completion_input_ids[s, prompt_len:]
                batch_completion_tokens.append(completion_tokens)

                # Decode to text
                completion_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
                batch_completions.append(completion_text)

            completions.append(batch_completions)
            completion_tokens_list.append(batch_completion_tokens)

        # Create attention masks for completions
        completion_attention_masks = []
        for batch_tokens in completion_tokens_list:
            batch_masks = []
            for tokens in batch_tokens:
                mask = torch.ones(len(tokens), device=device)
                batch_masks.append(mask)
            completion_attention_masks.append(batch_masks)

        # Extract logits for computing loss
        logits = generation_output.scores if hasattr(generation_output, 'scores') else []

        return {
            "prompts": prompts,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "completions": completions,
            "completion_input_ids": completion_tokens_list,
            "completion_attention_mask": completion_attention_masks,
            "logits": logits
        }

    def _compute_rewards(self, prompts, completions_list):
        """Compute rewards based on completions from agents, with TTR metrics logging when applicable."""
        if self.num_agents == 2 and callable(self.reward_funcs):

            # Only attempt to get TTR calculator if using vocabulary richness reward
            calculate_ttr = None
            if self.reward_funcs.__name__ == 'vocabulary_richness_reward':
                # Get reference to the calculate_ttr function
                # This assumes the reward function has this attribute set
                calculate_ttr = getattr(self.reward_funcs, 'calculate_ttr', None)

            # Single prompt with multiple generations per agent
            if len(prompts) == 1:
                # Ensure correct structure
                if not isinstance(completions_list[0], list) or not isinstance(completions_list[1], list):
                    self.logger.error(
                        f"Expected lists of completions, got: {type(completions_list[0])}, {type(completions_list[1])}")
                    completions_list = [
                        [completions_list[0]] if not isinstance(completions_list[0], list) else completions_list[0],
                        [completions_list[1]] if not isinstance(completions_list[1], list) else completions_list[1]
                    ]

                self.logger.info(
                    f"Processing {len(completions_list[0])} completions from agent 1 and {len(completions_list[1])} from agent 2")

                all_rewards = []
                min_completions = min(len(completions_list[0]), len(completions_list[1]))

                for i in range(min_completions):
                    completion1 = completions_list[0][i]
                    completion2 = completions_list[1][i]
                    len1 = len(completion1)
                    len2 = len(completion2)

                    # Log metrics
                    if self.wandb_initialized and i == 0:  # Just log the first pair
                        log_data = {
                            "agent1_completion_length": len1,
                            "agent2_completion_length": len2,
                            "max_min_len_ratio": max(len1, len2) / min(len1, len2) if min(len1, len2) > 0 else max(len1,
                                                                                                                   len2),
                            "agent_2_1_len_ratio": len2 / len1 if len1 > 0 else len2,
                        }

                        # Add TTR metrics only if we have the calculator
                        if calculate_ttr:
                            ttr1 = calculate_ttr(completion1, STOPWORDS)
                            ttr2 = calculate_ttr(completion2, STOPWORDS)
                            log_data.update({
                                "agent1_ttr": ttr1,
                                "agent2_ttr": ttr2,
                                "ttr_ratio": ttr2 / ttr1 if ttr1 > 0 else float('inf'),
                                "ttr_improvement": max(0, ttr2 - ttr1),
                            })

                        wandb.log(log_data)

                    # Call reward function and collect rewards
                    pair_reward = self.reward_funcs([completion1], [completion2])
                    all_rewards.extend(pair_reward)

                return all_rewards
            else:
                # Batch processing (multiple prompts)
                agent1_completions = []
                agent2_completions = []

                # Extract completions
                for prompt_idx in range(len(prompts)):
                    if prompt_idx < len(completions_list[0]):
                        agent1_completion = completions_list[0][prompt_idx][0] if isinstance(
                            completions_list[0][prompt_idx], list) else completions_list[0][prompt_idx]
                        agent1_completions.append(agent1_completion)

                    if prompt_idx < len(completions_list[1]):
                        agent2_completion = completions_list[1][prompt_idx][0] if isinstance(
                            completions_list[1][prompt_idx], list) else completions_list[1][prompt_idx]
                        agent2_completions.append(agent2_completion)

                # Log TTR metrics for batch processing if applicable
                if self.wandb_initialized and calculate_ttr and len(agent1_completions) > 0 and len(
                        agent2_completions) > 0:
                    ttr1_values = [calculate_ttr(completion, STOPWORDS) for completion in agent1_completions]
                    ttr2_values = [calculate_ttr(completion, STOPWORDS) for completion in agent2_completions]

                    avg_ttr1 = sum(ttr1_values) / len(ttr1_values)
                    avg_ttr2 = sum(ttr2_values) / len(ttr2_values)

                    wandb.log({
                        "avg_agent1_ttr": avg_ttr1,
                        "avg_agent2_ttr": avg_ttr2,
                        "avg_ttr_ratio": avg_ttr2 / avg_ttr1 if avg_ttr1 > 0 else float('inf'),
                    })

                # Call the reward function
                return self.reward_funcs(agent1_completions, agent2_completions)
        else:
            raise NotImplementedError(
                "Currently, only the case with 2 agents and a callable reward function is implemented."
            )

    def _compute_loss_with_gradients(self, agent, completions_data, rewards):
        """
        Compute loss with proper gradient tracking by performing a new forward pass.

        Args:
            agent (PreTrainedModel): The agent model.
            completions_data (dict): The completions data from _generate_completions.
            rewards (List[float]): The rewards for each completion.

        Returns:
            torch.Tensor: The computed loss with gradients attached.
        """
        device = agent.device

        # Make sure we have the correct number of rewards
        if len(rewards) == 0:
            self.logger.warning("No rewards provided for loss computation")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)

        # Use baseline approach
        rewards_baseline = rewards_tensor.mean()  # Use mean as baseline
        advantages = rewards_tensor - rewards_baseline  # Compute advantages

        # Clip advantages to reasonable range to prevent numerical instability
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        # Set agent to train mode to ensure gradients are tracked
        agent.train()

        prompt_input_ids = completions_data["prompt_input_ids"]
        prompt_attention_mask = completions_data["prompt_attention_mask"]
        completion_input_ids = completions_data["completion_input_ids"]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        # Process each prompt in the batch
        for batch_idx in range(len(prompt_input_ids)):
            prompt_ids = prompt_input_ids[batch_idx]
            prompt_mask = prompt_attention_mask[batch_idx]

            # Process each generated completion for this prompt
            for seq_idx, completion_tokens in enumerate(completion_input_ids[batch_idx]):
                # Break if we've processed enough completions for the available rewards
                if seq_idx >= len(advantages):
                    break

                advantage = advantages[seq_idx]

                # Create input sequence by concatenating prompt with all but last token of completion
                # (we'll predict the next token at each step)
                if len(completion_tokens) > 0:
                    input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])

                    # Target is the completion tokens
                    target_ids = completion_tokens

                    # Create attention mask for the full sequence
                    attention_mask = torch.ones(len(input_ids), device=device)

                    # Forward pass with gradients enabled
                    outputs = agent(
                        input_ids=input_ids.unsqueeze(0),  # Add batch dimension
                        attention_mask=attention_mask.unsqueeze(0),  # Add batch dimension
                    )

                    # Get logits for the completion part (excluding prompt)
                    completion_logits = outputs.logits[0, prompt_ids.size(0) - 1:-1, :]

                    # Calculate log probabilities
                    log_probs = []
                    for i, token_id in enumerate(target_ids):
                        if i < completion_logits.size(0):  # Check if we have logits for this position
                            token_logits = completion_logits[i]
                            token_log_prob = torch.log_softmax(token_logits, dim=-1)[token_id]
                            log_probs.append(token_log_prob)

                    if log_probs:
                        sequence_log_prob = torch.stack(log_probs).sum()
                        # Policy gradient loss: -log_prob * advantage
                        loss = -sequence_log_prob * advantage
                        total_loss = total_loss + loss
                        num_samples += 1

        # Average the loss over all processed samples
        if num_samples > 0:
            total_loss = total_loss / num_samples

        # Safety check for invalid loss values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.logger.warning(f"Invalid loss detected: {total_loss}, using default loss")
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss

    def _compute_loss_with_kl_penalty(self, agent, completions_data, rewards, old_model=None, kl_coef=0.1):
        """
        Compute loss with proper gradient tracking and KL divergence regularization.

        Args:
            agent (PreTrainedModel): The agent model.
            completions_data (dict): The completions data from _generate_completions.
            rewards (List[float]): The rewards for each completion.
            old_model (PreTrainedModel, optional): The old model state for KL calculation.
            kl_coef (float): Coefficient for KL penalty term.

        Returns:
            torch.Tensor: The computed loss with gradients attached.
        """
        device = agent.device

        # Base loss calculation - similar to _compute_loss_with_gradients
        # Make sure we have the correct number of rewards
        if len(rewards) == 0:
            self.logger.warning("No rewards provided for loss computation")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)

        # Use baseline approach
        rewards_baseline = rewards_tensor.mean()  # Use mean as baseline
        advantages = rewards_tensor - rewards_baseline  # Compute advantages

        # Clip advantages to reasonable range to prevent numerical instability
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        # Set agent to train mode to ensure gradients are tracked
        agent.train()

        prompt_input_ids = completions_data["prompt_input_ids"]
        prompt_attention_mask = completions_data["prompt_attention_mask"]
        completion_input_ids = completions_data["completion_input_ids"]

        # Initialize policy gradient loss
        pg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # Initialize KL loss
        kl_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        # Process each prompt in the batch
        for batch_idx in range(len(prompt_input_ids)):
            prompt_ids = prompt_input_ids[batch_idx]
            prompt_mask = prompt_attention_mask[batch_idx]

            # Process each generated completion for this prompt
            for seq_idx, completion_tokens in enumerate(completion_input_ids[batch_idx]):
                # Break if we've processed enough completions for the available rewards
                if seq_idx >= len(advantages):
                    break

                advantage = advantages[seq_idx]

                # Process only if we have completion tokens
                if len(completion_tokens) > 0:
                    input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
                    target_ids = completion_tokens
                    attention_mask = torch.ones(len(input_ids), device=device)

                    # Get current policy distribution
                    outputs = agent(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                    )
                    current_logits = outputs.logits[0, prompt_ids.size(0) - 1:-1, :]

                    # Calculate log probabilities for policy gradient
                    log_probs = []
                    for i, token_id in enumerate(target_ids):
                        if i < current_logits.size(0):
                            token_logits = current_logits[i]
                            token_log_prob = torch.log_softmax(token_logits, dim=-1)[token_id]
                            log_probs.append(token_log_prob)

                    # Calculate policy gradient loss component
                    if log_probs:
                        sequence_log_prob = torch.stack(log_probs).sum()
                        # Policy gradient loss: -log_prob * advantage
                        pg_loss = pg_loss - sequence_log_prob * advantage
                        num_samples += 1

                    # Add KL divergence term if old_model is provided
                    if old_model is not None:
                        old_model.eval()
                        with torch.no_grad():  # No gradients for old model
                            old_outputs = old_model(
                                input_ids=input_ids.unsqueeze(0),
                                attention_mask=attention_mask.unsqueeze(0),
                            )
                            old_logits = old_outputs.logits[0, prompt_ids.size(0) - 1:-1, :]

                        # Calculate KL divergence at each token position
                        for i in range(min(old_logits.size(0), current_logits.size(0))):
                            old_probs = torch.softmax(old_logits[i], dim=-1)
                            current_probs = torch.softmax(current_logits[i], dim=-1)
                            # Small epsilon to prevent log(0)
                            eps = 1e-8
                            # KL divergence: sum(p_old * log(p_old / p_new))
                            token_kl = torch.sum(
                                old_probs * (torch.log(old_probs + eps) - torch.log(current_probs + eps)))
                            kl_loss = kl_loss + token_kl

        # Average the loss components
        if num_samples > 0:
            pg_loss = pg_loss / num_samples
            if old_model is not None:
                kl_loss = kl_loss / num_samples

        # Total loss with KL penalty
        total_loss = pg_loss
        if old_model is not None:
            total_loss = total_loss + kl_coef * kl_loss

            # Log KL loss to wandb if enabled
            if self.wandb_initialized:
                wandb.log({
                    "pg_loss": pg_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "kl_penalty": (kl_coef * kl_loss).item()
                })

        # Safety check for invalid loss values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.logger.warning(f"Invalid loss detected: {total_loss}, using default loss")
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss

    def _log_training_progress(self, epoch, batch_idx, rewards):
        """
        Log training progress.

        Args:
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
            rewards (List[float]): Rewards for the current batch.
        """
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        metrics = {
            "epoch": epoch,
            "batch": batch_idx,
            "average_reward": avg_reward,
        }

        self.logger.info(f"Training progress: {metrics}")

    def _save_checkpoints(self, epoch, batch_idx):
        """
        Save model checkpoints.

        Args:
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
        """
        output_dir = self.args.output_dir
        if not output_dir:
            self.logger.warning("Output directory not specified. Skipping checkpoint saving.")
            return

        os.makedirs(output_dir, exist_ok=True)

        for agent_idx, agent in enumerate(self.agents):
            agent_dir = f"{output_dir}/agent_{agent_idx}/epoch_{epoch}_batch_{batch_idx}"
            os.makedirs(agent_dir, exist_ok=True)

            # Save agent state
            agent.save_pretrained(agent_dir)

            # Save tokenizer if available
            if self.tokenizer:
                self.tokenizer.save_pretrained(agent_dir)

            # Save optimizer state
            torch.save(self.optimizers[agent_idx].state_dict(),
                       f"{agent_dir}/optimizer.pt")

        self.logger.info(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}")

    def _load_from_checkpoint(self, checkpoint_dir):
        """
        Load models and training state from checkpoint.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.

        Returns:
            int: The epoch to resume from.
        """
        # Extract epoch from checkpoint path
        import re
        epoch_match = re.search(r'epoch_(\d+)', checkpoint_dir)
        epoch = int(epoch_match.group(1)) if epoch_match else 0

        for agent_idx, agent in enumerate(self.agents):
            agent_dir = f"{checkpoint_dir}/agent_{agent_idx}"

            # Load model weights
            agent.load_state_dict(torch.load(f"{agent_dir}/pytorch_model.bin"))

            # Load optimizer state
            optimizer_path = f"{agent_dir}/optimizer.pt"
            if os.path.exists(optimizer_path):
                self.optimizers[agent_idx].load_state_dict(torch.load(optimizer_path))

        self.logger.info(f"Resumed training from checkpoint: {checkpoint_dir}")
        return epoch + 1  # Resume from next epoch

    def save_model(self, output_dir):
        """
        Save the final trained models.

        Args:
            output_dir (str): Directory to save the models to.
        """
        os.makedirs(output_dir, exist_ok=True)

        for agent_idx, agent in enumerate(self.agents):
            agent_dir = f"{output_dir}/agent_{agent_idx}"
            os.makedirs(agent_dir, exist_ok=True)

            agent.save_pretrained(agent_dir)

            if self.tokenizer:
                self.tokenizer.save_pretrained(agent_dir)

        self.logger.info(f"All agent models saved to {output_dir}")

        # Log final model saving to wandb
        if self.wandb_initialized:
            wandb.log({
                "final_model_saved": output_dir
            })

            # Log model differences - e.g. how much did the agents diverge?
            if self.num_agents >= 2:
                # Sample a short fixed prompt to compare agent outputs
                sample_prompt = "Write a short response to the following: Hello, how are you?"
                sample_outputs = []

                # Generate a sample output from each agent
                for agent_idx, agent in enumerate(self.agents):
                    agent.eval()
                    with torch.no_grad():
                        inputs = self.tokenizer(sample_prompt, return_tensors="pt").to(agent.device)
                        output = agent.generate(
                            **inputs,
                            max_length=50,
                            do_sample=True,
                            temperature=0.7
                        )
                        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                        sample_outputs.append(decoded)

                # Log the sample outputs
                for agent_idx, output in enumerate(sample_outputs):
                    wandb.log({
                        f"final_agent{agent_idx + 1}_sample_output": output
                    })

# Define a simple external reward function
def length_ratio_reward(completions1, completions2):
    """Example reward function that rewards based on length ratio between agent outputs"""
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        len1, len2 = len(c1), len(c2)
        # Reward based on the ratio of lengths
        ratio = max(len1, len2) / min(len1, len2) if min(len1, len2) > 0 else max(len1, len2)
        rewards.append(float(ratio))
    return rewards


def proper_length_ratio_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion is 2-3 times longer than the first.

    The maximum reward is given when the ratio is exactly in the target range (2-3x),
    and gradually decreases as the ratio moves further from this range.
    """
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        len1, len2 = len(c1), len(c2)

        # Ensure we don't divide by zero
        if len1 == 0:
            rewards.append(0.0)  # No reward for empty first completion
            continue

        # Calculate the ratio of second to first completion
        ratio = len2 / len1

        # Define target range and calculate reward
        target_min = 2.0
        target_max = 3.0

        if target_min <= ratio <= target_max:
            # Maximum reward (1.0) when within the target range
            reward = 1.0
        else:
            # Calculate distance from the nearest boundary of the target range
            if ratio < target_min:
                distance = target_min - ratio
            else:  # ratio > target_max
                distance = ratio - target_max

            # Reward decreases as distance increases
            # Using an exponential decay function: reward = e^(-distance)
            import math
            reward = math.exp(-distance)

        rewards.append(float(reward))

    return rewards


def vocabulary_richness_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion has higher
    vocabulary richness (Type-Token Ratio without stopwords) than the first.

    The reward is based on the improvement in TTR from the first to the second completion.
    Maximum reward is given when the second completion's TTR is substantially higher,
    and gradually decreases as the improvement diminishes.
    """
    import math

    def calculate_ttr(text, stopwords):
        """Calculate Type-Token Ratio (TTR) excluding stopwords.

        Args:
            text: String text to analyze
            stopwords: Set of stopwords to exclude

        Returns:
            Float value representing TTR (unique content words / total content words)
        """
        import re

        # Tokenize by splitting on non-alphanumeric characters and convert to lowercase
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out stopwords
        if stopwords:
            content_words = [word for word in words if word not in stopwords]
        else:
            content_words = words

        # Calculate TTR (unique words / total words)
        if not content_words:
            return 0.0

        types = len(set(content_words))
        tokens = len(content_words)

        return types / tokens if tokens > 0 else 0.0

    vocabulary_richness_reward.calculate_ttr = calculate_ttr
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Calculate TTR for both completions
        ttr1 = calculate_ttr(c1, STOPWORDS)
        ttr2 = calculate_ttr(c2, STOPWORDS)

        # Handle edge cases
        if ttr1 == 0:
            if ttr2 > 0:
                reward = 1.0  # Maximum reward if improvement from zero
            else:
                reward = 0.0  # No reward if both are zero
        else:
            # Calculate improvement ratio
            improvement = ttr2 / ttr1

            # Define target range for improvement
            target_min = 1.2  # At least 20% improvement
            target_max = 2.0  # Up to double the vocabulary richness

            if improvement >= target_max:
                reward = 1.0  # Maximum reward
            elif improvement >= target_min:
                # Linear scaling between min and max targets
                reward = (improvement - target_min) / (target_max - target_min)
            else:
                # Exponential decay for below-target improvement
                distance = target_min - improvement
                reward = math.exp(-2 * distance)  # Steeper decay

        rewards.append(float(reward))

    return rewards


def example_usage():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    # Load tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure MAGRPO
    config = MAGRPOConfig(
        output_dir="./magrpo_lora_output",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        num_generations=8,
        max_new_tokens=256,
    )

    # Create dataset
    from datasets import Dataset
    from datasets import load_dataset
    # train_data = {
    #     "prompt": [
    #         "Write a story about a robot:",
    #         "Explain quantum physics:",
    #         "Create a recipe for chocolate cake:",
    #         "Write a song about snow:",
    #         "Describe a city in the clouds:",
    #         "Invent a new holiday and explain it:",
    #         "Write a bedtime story for a dragon:",
    #         "Explain how teleportation might work:",
    #         "Design a futuristic bicycle:",
    #         "Tell a joke about dinosaurs:",
    #         "Write a poem about the ocean at night:",
    #         "Describe a world without electricity:",
    #         "Create a superhero with a unique power:",
    #         "Write a scene where the moon talks:",
    #         "Explain black holes to a 5-year-old:",
    #         "Invent a new type of fruit:",
    #         "Describe the life of a time-traveling cat:",
    #         "Write an apology letter from a ghost:",
    #         "Explain gravity using a pizza analogy:",
    #         "Design a playground on Mars:",
    #         "Write a love letter between two stars:",
    #         "Invent a game played by aliens:",
    #         "Describe a school for magical creatures:",
    #         "Write a recipe for an invisible soup:",
    #         "Explain Wi-Fi to someone from the 1800s:",
    #         "Create a workout plan for robots:",
    #         "Describe a hotel at the bottom of the ocean:",
    #         "Write a story about a lost shadow:",
    #         "Invent a musical instrument from glass:",
    #         "Explain the internet using only food terms:",
    #         "Design a zoo for extinct animals:",
    #         "Write a diary entry from a raindrop:",
    #         "Describe a world where pets can talk:",
    #         "Explain how dreams are made:",
    #         "Create a menu for a restaurant in space:",
    #         "Write a letter from a tree to a human:",
    #         "Invent a holiday where everyone wears pajamas:",
    #         "Describe a rainbow factory:",
    #         "Write a scene from a robot cooking show:",
    #         "Explain the weather like a pirate would:"
    #     ]
    # }
    # train_dataset = Dataset.from_dict(train_data)

    dataset_name = "trl-lib/tldr"
    dataset_split = "train[:100]"
    train_dataset = load_dataset(dataset_name, split=dataset_split)

    # Configure wandb
    wandb_config = {
        "project": "trl",
        "entity": "nu-llpr",
        "name": "qwen2.5-0.5B-lora-magrpo",
    }

    # Configure LoRA
    lora_config = LoraConfig(
        r=1024,  # Increased rank for better capacity/expressivity
        lora_alpha=2048,  # Increased alpha to maintain same alpha/r ratio (2:1)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More modules
        lora_dropout=0.1,  # Slightly higher dropout for better regularization
        bias="none",  # Keep bias fixed to prevent overfitting
        modules_to_save=["embed_tokens", "lm_head"],  # Tune embedding and output layers
        fan_in_fan_out=False,  # Set appropriately based on model architecture
        task_type=TaskType.CAUSAL_LM
    )
    # lora_config = LoraConfig(
    #     r=32,  # Increased rank for better capacity/expressivity
    #     lora_alpha=64,  # Increased alpha to maintain same alpha/r ratio (2:1)
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #     lora_dropout=0.05,  # Slightly higher dropout for better regularization
    #     bias="none",  # Keep bias fixed to prevent overfitting
    #     fan_in_fan_out=False,  # Set appropriately based on model architecture
    #     task_type=TaskType.CAUSAL_LM
    # )

    # Create agents list with two independent LoRA models
    agents = []
    for _ in range(2):
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.print_trainable_parameters()
        # lora_model = base_model
        agents.append(lora_model)

    # Initialize trainer with our pre-created agents
    trainer = MAGRPOTrainer(
        agents=agents,
        reward_funcs=vocabulary_richness_reward,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        wandb_config=wandb_config,
    )

    # Train
    trainer.train()

    # Save models
    for i, agent in enumerate(trainer.agents):
        agent.save_pretrained(f"{config.output_dir}/final_lora_agent_{i}")
    tokenizer.save_pretrained(f"{config.output_dir}/tokenizer")
    print("Training complete!")


if __name__ == "__main__":
    example_usage()