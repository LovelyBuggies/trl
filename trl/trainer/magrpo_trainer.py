
import torch
from typing import Callable, List, Optional, Union, Dict, Any, Tuple
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from trl import MAGRPOConfig
import numpy as np
import logging
import os
import wandb
from trl import STOPWORDS, vocabulary_richness_reward, proper_length_ratio_reward, \
        sentiment_contrast_reward, syntax_complexity_reward, readability_contrast_reward, question_generation_reward, \
        fact_density_reward, coherence_reward, summarization_reward

RewardFunc = Union[str, PreTrainedModel, Callable[[List[str], List[str]], List[float]]]


class MAGRPOTrainer:
    """
    Multi-Agent Group Relative Policy Optimization Trainer (MAGRPO).
    Currently, only supports homogenous agents and shared reward functions.

    Args:
        model (str or PreTrainedModel): The model to be trained for homogenous agents
        num_agents (int): The number of agents.
        reward_funcs (RewardFunc or list[RewardFunc]): The reward functions for all agents.
        reward_weights (list[float], optional): The weights for each reward function.
        reward_processors (list[Callable], optional): Processors to apply to rewards (e.g., scaling).
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
            reward_weights: Optional[List[float]] = None,
            reward_processors: Optional[List[Callable]] = None,
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

        # Set up reward functions
        self._setup_reward_functions(reward_funcs, reward_weights, reward_processors)

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

    def _setup_reward_functions(self, reward_funcs, reward_weights=None, reward_processors=None):
        """Set up reward functions with weights and processors."""
        # Convert single reward function to list for uniform handling
        if not isinstance(reward_funcs, list):
            self.reward_funcs = [reward_funcs]
        else:
            self.reward_funcs = reward_funcs

        # Set up reward weights (default to equal weights if not provided)
        if reward_weights is None:
            self.reward_weights = [1.0 / len(self.reward_funcs)] * len(self.reward_funcs)
        else:
            if len(reward_weights) != len(self.reward_funcs):
                raise ValueError(f"Number of reward weights ({len(reward_weights)}) must match "
                                 f"number of reward functions ({len(self.reward_funcs)})")
            # Normalize weights to sum to 1
            total = sum(reward_weights)
            self.reward_weights = [w / total for w in reward_weights]

        # Set up reward processors
        if reward_processors is None:
            # Default identity processor
            self.reward_processors = [lambda x: x] * len(self.reward_funcs)
        else:
            if len(reward_processors) != len(self.reward_funcs):
                raise ValueError(f"Number of reward processors ({len(reward_processors)}) must match "
                                 f"number of reward functions ({len(self.reward_funcs)})")

            # Handle None processors by replacing with identity function
            self.reward_processors = []
            for processor in reward_processors:
                if processor is None:
                    self.reward_processors.append(lambda x: x)  # Identity function
                else:
                    self.reward_processors.append(processor)

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
                "num_reward_functions": len(self.reward_funcs),
                "reward_weights": self.reward_weights,
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "num_train_epochs": self.args.num_train_epochs,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,
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
            # Track individual reward components
            epoch_reward_components = [[] for _ in range(len(self.reward_funcs))]

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
                                max_new_tokens=self.args.max_new_tokens,
                                **kwargs
                            )
                            all_completions.append(agent_completions)

                        # Extract completions for reward calculation
                        agent_completions_list = []
                        for agent_idx in range(self.num_agents):
                            agent_completions_list.append(all_completions[agent_idx]["completions"][0])

                        # Compute rewards based on all agents' completions for this prompt
                        rewards, reward_components = self._compute_rewards([prompt], agent_completions_list)
                        epoch_rewards.extend(rewards)

                        # Track reward components
                        for i, component in enumerate(reward_components):
                            epoch_reward_components[i].extend(component)

                        # Track rewards per agent for more detailed logging
                        for agent_idx in range(self.num_agents):
                            for reward in rewards:
                                epoch_agent_rewards[agent_idx].append(reward)

                        # Update each agent using the rewards with proper gradient tracking
                        batch_loss = 0.0
                        agent_losses = []
                        for agent_idx in range(self.num_agents):
                            # Compute loss with the new function that enables proper gradient tracking
                            agent_loss = self._compute_loss_with_gradients(
                                self.agents[agent_idx],
                                all_completions[agent_idx],
                                rewards
                            )

                            # Backward pass and optimization
                            agent_loss.backward()
                            self.optimizers[agent_idx].step()

                            batch_loss += agent_loss.detach().item()
                            agent_losses.append(agent_loss.detach().item())

                        epoch_loss += batch_loss

                        # Log to wandb per batch
                        if self.wandb_initialized:
                            log_data = {
                                "batch_loss": batch_loss,
                                "batch_rewards_mean": np.mean(rewards) if rewards else 0,
                                "step": epoch * len(self.get_train_dataloader()) + batch_idx,
                            }

                            # Log individual agent losses
                            for i, loss in enumerate(agent_losses):
                                log_data[f"agent{i + 1}_loss"] = loss

                            # Log individual reward components
                            for i, component in enumerate(reward_components):
                                component_mean = np.mean(component) if component else 0
                                log_data[f"reward_{i + 1}_mean"] = component_mean

                            wandb.log(log_data)

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
            # Calculate average reward components
            avg_reward_components = [sum(comp) / len(comp) if comp else 0 for comp in epoch_reward_components]

            epoch_log = {
                "epoch": epoch,
                "epoch_loss": epoch_loss / len(self.get_train_dataloader()) if epoch_loss else 0,
                "epoch_avg_reward": avg_reward,
            }

            # Add agent-specific reward tracking
            for i, avg_agent_reward in enumerate(avg_agent_rewards):
                epoch_log[f"agent{i + 1}_avg_reward"] = avg_agent_reward

            # Add component-specific reward tracking
            for i, avg_component in enumerate(avg_reward_components):
                epoch_log[f"reward_{i + 1}_avg"] = avg_component

            if self.wandb_initialized:
                wandb.log(epoch_log)

            # Prepare log message
            log_message = f"Epoch {epoch}: Loss={epoch_loss / len(self.get_train_dataloader())}, " \
                          f"Avg Reward={avg_reward}"

            # Add agent rewards to log message
            for i, avg_agent_reward in enumerate(avg_agent_rewards):
                log_message += f", Agent{i + 1} Avg Reward={avg_agent_reward}"

            # Add component rewards to log message
            for i, avg_component in enumerate(avg_reward_components):
                log_message += f", Reward{i + 1} Avg={avg_component}"

            self.logger.info(log_message)

        # Close wandb
        if self.wandb_initialized:
            wandb.finish()

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

    def _compute_rewards(self, prompts, completions_list) -> Tuple[List[float], List[List[float]]]:
        """
        Compute combined rewards based on multiple reward functions, with weights.

        Args:
            prompts: List of prompts
            completions_list: List of completions from each agent

        Returns:
            Tuple containing:
            - List of final weighted rewards
            - List of individual reward components (for logging)
        """
        if self.num_agents != 2:
            raise NotImplementedError(
                "Currently, only the case with 2 agents is implemented."
            )

        # Initialize lists to store rewards
        all_rewards = []
        all_reward_components = [[] for _ in range(len(self.reward_funcs))]

        # Single prompt case
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

            min_completions = min(len(completions_list[0]), len(completions_list[1]))

            for i in range(min_completions):
                completion1 = completions_list[0][i]
                completion2 = completions_list[1][i]

                # Log metrics for the first pair if wandb is enabled
                if self.wandb_initialized and i == 0:
                    self._log_completion_metrics(completion1, completion2)

                # Calculate rewards from each function and apply weights
                weighted_reward = 0.0
                reward_components = []

                for func_idx, (reward_func, weight, processor) in enumerate(
                        zip(self.reward_funcs, self.reward_weights, self.reward_processors)
                ):
                    # Call reward function for this pair
                    pair_rewards = reward_func([completion1], [completion2])

                    # Apply processor to rewards
                    processed_rewards = [processor(r) for r in pair_rewards]

                    # Store the raw component rewards for logging
                    reward_components.append(processed_rewards[0])
                    all_reward_components[func_idx].extend(processed_rewards)

                    # Add weighted component to total reward
                    weighted_reward += weight * processed_rewards[0]

                all_rewards.append(weighted_reward)

            return all_rewards, all_reward_components
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

            # Log metrics for batch if wandb is enabled
            if self.wandb_initialized and len(agent1_completions) > 0 and len(agent2_completions) > 0:
                self._log_batch_metrics(agent1_completions, agent2_completions)

            # Calculate rewards for each function
            weighted_rewards = [0.0] * len(agent1_completions)

            for func_idx, (reward_func, weight, processor) in enumerate(
                    zip(self.reward_funcs, self.reward_weights, self.reward_processors)
            ):
                # Call reward function for all samples
                batch_rewards = reward_func(agent1_completions, agent2_completions)

                # Apply processor to rewards
                processed_rewards = [processor(r) for r in batch_rewards]

                # Store component rewards for logging
                all_reward_components[func_idx].extend(processed_rewards)

                # Add weighted component to total rewards
                for i, r in enumerate(processed_rewards):
                    if i < len(weighted_rewards):
                        weighted_rewards[i] += weight * r

            return weighted_rewards, all_reward_components

    def _log_completion_metrics(self, completion1, completion2):
        """Log detailed metrics about a pair of completions to wandb."""

        calculate_len_ratio = None
        for func in self.reward_funcs:
            if hasattr(func, '__name__') and func.__name__ == 'proper_length_ratio_reward':
                calculate_len_ratio = True
                break

        calculate_ttr = None
        for func in self.reward_funcs:
            if hasattr(func, '__name__') and func.__name__ == 'vocabulary_richness_reward':
                calculate_ttr = getattr(func, 'calculate_ttr', None)
                break

        log_data = {}
        if calculate_len_ratio:
            len1 = len(completion1)
            len2 = len(completion2)
            log_data.update({
                "agent1_completion_length": len1,
                "agent2_completion_length": len2,
                "agents_len_ratio": len2 / len1 if len1 > 0 else len2,
            })

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

    def _log_batch_metrics(self, agent1_completions, agent2_completions):
        """Log aggregate metrics about batches of completions to wandb."""
        # Log length metrics
        agent1_lengths = [len(c) for c in agent1_completions]
        agent2_lengths = [len(c) for c in agent2_completions]

        avg_len1 = sum(agent1_lengths) / len(agent1_lengths) if agent1_lengths else 0
        avg_len2 = sum(agent2_lengths) / len(agent2_lengths) if agent2_lengths else 0

        log_data = {
            "avg_agent1_completion_length": avg_len1,
            "avg_agent2_completion_length": avg_len2,
            "avg_length_ratio": avg_len2 / avg_len1 if avg_len1 > 0 else float('inf'),
        }

        # Add TTR metrics if available
        calculate_ttr = None
        for func in self.reward_funcs:
            if hasattr(func, '__name__') and func.__name__ == 'vocabulary_richness_reward':
                calculate_ttr = getattr(func, 'calculate_ttr', None)
                break

        if calculate_ttr:
            ttr1_values = [calculate_ttr(completion, STOPWORDS) for completion in agent1_completions]
            ttr2_values = [calculate_ttr(completion, STOPWORDS) for completion in agent2_completions]

            avg_ttr1 = sum(ttr1_values) / len(ttr1_values) if ttr1_values else 0
            avg_ttr2 = sum(ttr2_values) / len(ttr2_values) if ttr2_values else 0

            log_data.update({
                "avg_agent1_ttr": avg_ttr1,
                "avg_agent2_ttr": avg_ttr2,
                "avg_ttr_ratio": avg_ttr2 / avg_ttr1 if avg_ttr1 > 0 else float('inf'),
            })

        wandb.log(log_data)

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


# Define reward processors that can be used with the MAGRPOTrainer
class RewardProcessors:
    """Collection of reward processing functions to modify raw rewards."""

    @staticmethod
    def identity():
        """Return an identity processor that returns the reward unchanged."""
        return lambda x: x

    @staticmethod
    def clamp(min_val=-10.0, max_val=10.0):
        """Return a processor that clamps rewards to a range."""
        return lambda x: max(min_val, min(max_val, x))

    @staticmethod
    def sigmoid_scale():
        """Return a processor that applies sigmoid scaling to rewards."""
        import math
        return lambda x: 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def normalize_by_length(max_len=1000):
        """Return a processor that normalizes rewards by text length."""

        def processor(reward, text_length=None):
            if text_length is None:
                return reward
            norm_factor = min(1.0, text_length / max_len)
            return reward * norm_factor

        return processor

    @staticmethod
    def exponential_scale(factor=1.0):
        """Return a processor that applies exponential scaling to rewards."""
        import math
        return lambda x: math.exp(factor * x) - 1 if x > 0 else -math.exp(-factor * x) + 1


# Example usage with multiple reward functions
def example_usage_multi_reward():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset

    # Load tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure MAGRPO
    config = MAGRPOConfig(
        output_dir="./magrpo_multi_reward_output",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        num_generations=8,
        max_new_tokens=256,
    )

    # Create dataset
    dataset_name = "trl-lib/tldr"
    dataset_split = "train[:100]"
    train_dataset = load_dataset(dataset_name, split=dataset_split)

    # Configure wandb
    wandb_config = {
        "project": "trl",
        "entity": "nu-llpr",
        "name": "qwen-magrpo-multi-reward",
    }

    # Set up reward functions with weights
    reward_funcs = [
        sentiment_contrast_reward,
    ]
    reward_weights = [1.0]

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

    # Create agents list with two independent LoRA models
    agents = []
    for _ in range(2):
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        # lora_model = get_peft_model(base_model, lora_config)
        # lora_model.print_trainable_parameters()
        lora_model = base_model
        agents.append(lora_model)

    # Initialize trainer with multiple reward functions
    trainer = MAGRPOTrainer(
        agents=agents,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        reward_processors=reward_processors,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        wandb_config=wandb_config,
    )

    # Train
    trainer.train()

    # Save models
    trainer.save_model(f"{config.output_dir}/final_models")
    print("Training complete!")


if __name__ == "__main__":
    example_usage_multi_reward()