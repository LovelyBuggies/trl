from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
import torch
import argparse
import yaml
import os

def parse_arguments():
    """Parse command line arguments for configuration settings."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with GRPO and different compression levels."
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--compression", 
        type=str, 
        default=None,
        choices=["high", "medium", "low"],
        help="Compression level: high (4-bit), medium (8-bit), or low (16-bit)"
    )
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(f"Error in configuration file: {exc}")
        return {}
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        return {}

def reward_length(completions, **kwargs):
    """Reward function that favors completions close to exactly 50 characters."""
    # target_length = 20
    # return [-abs(target_length - len(completion)) for completion in completions]
    # duel reward
    target_length = 50
    rewards = []
    for completion in completions:
        length = len(completion)
        # Calculate how close the completion is to the target length
        # The closer to target_length, the higher the reward (max 1.0)
        distance = abs(length - target_length)
        # Use exponential decay to create a reward that drops off quickly as we move away from target
        reward = max(0.0, 1.0 - (distance / target_length))
        rewards.append(10 * reward)
    return rewards

def reward_capitalization(completions, **kwargs):
    """Reward function based on the percentage of capital letters in the response."""
    rewards = []
    for completion in completions:
        # Count only alphabetic characters
        alpha_chars = [c for c in completion if c.isalpha()]
        
        if not alpha_chars:  # Avoid division by zero
            rewards.append(0.0)
            continue
            
        # Count capital letters
        capital_chars = [c for c in alpha_chars if c.isupper()]
        
        # Calculate percentage of capital letters (0.0 to 1.0)
        capital_percentage = len(capital_chars) / len(alpha_chars)
        
        # The reward is directly proportional to the percentage of capital letters
        rewards.append(10 * capital_percentage)
    
    return rewards

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load config from YAML file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Command line arguments override config file settings
    compression = args.compression or config.get('compression', 'high')
    
    # Get other parameters from config
    use_wandb = config.get('use_wandb', False)
    wandb_project = config.get('wandb_project', 'language-model-training')
    wandb_entity = config.get('wandb_entity', None)
    run_name = config.get('run_name', f'qwen-0.5b-grpo-{compression}-compression')
    
    print(f"Using {compression} compression level")
    if use_wandb:
        # Initialize wandb if enabled
        if wandb_entity:
            print(f"Weights & Biases enabled for entity: {wandb_entity}")
        else:
            print(f"Weights & Biases enabled (default entity)")
        print(f"Project: {wandb_project}")
        print(f"Run name: {run_name}")
        
        # Import and initialize wandb here if it's enabled
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config={
                        "compression": compression,
                        "model_name": config.get('model_name', "Qwen/Qwen2.5-0.5B"),
                        "learning_rate": config.get('learning_rate', 5e-5),
                        "epochs": config.get('num_train_epochs', 3),
                        "batch_size": config.get('batch_size', 4),
                    }
                )
            except ImportError:
                print("Warning: wandb not installed. Running without Weights & Biases tracking.")
                use_wandb = False
    
    # Load the dataset
    dataset_name = config.get('dataset', "trl-lib/tldr")
    dataset_split = config.get('dataset_split', "train")
    dataset = load_dataset(dataset_name, split=dataset_split)
    
    # Configure quantization based on compression level
    if compression == "high":
        print("Using high compression (4-bit quantization)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        # For high compression, we might want more aggressive LoRA settings
        lora_r = 8
        lora_alpha = 16
    elif compression == "medium":
        print("Using medium compression (8-bit quantization)")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        # For medium compression, we can use standard LoRA settings
        lora_r = 16
        lora_alpha = 32
    else:  # "low" or any other value
        print("Using low compression (16-bit/half precision)")
        bnb_config = None
        # For low compression, we can use more expressive LoRA settings
        lora_r = 32
        lora_alpha = 64
    
    # Load the model with appropriate quantization
    model_name = config.get('model_name', "Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA - adjust target modules as needed for your specific model
    target_modules = config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        # Additional memory optimizations:
        fan_in_fan_out=False,
        modules_to_save=None,
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # Print percentage of trainable parameters
    
    # Configure the GRPO trainer
    training_args = GRPOConfig(
        output_dir=config.get('output_dir', './results'),
        num_train_epochs=config.get('num_train_epochs', 3),
        per_device_train_batch_size=config.get('batch_size', 4),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        learning_rate=float(config.get('learning_rate', 5e-5)),
        max_grad_norm=config.get('max_grad_norm', 0.3),
        warmup_steps=config.get('warmup_steps', 100),
        logging_dir=config.get('logging_dir', './logs'),
        logging_steps=config.get('logging_steps', 10),
        save_steps=config.get('save_steps', 500),
        save_total_limit=config.get('save_total_limit', 3),
        
        # GRPO specific parameters
        max_prompt_length=config.get('max_prompt_length', 512),
        max_completion_length=config.get('max_completion_length', 128),
        num_generations=config.get('num_generations', 8),
        temperature=config.get('temperature', 1.0),
        top_p=config.get('top_p', 0.9),
        reward_weights=config.get('reward_weights', None),  # Moved reward_weights here
        
        # Reporting
        report_to="wandb" if use_wandb else "none",
        run_name=run_name if use_wandb else None,
    )
    
    # Configure reward functions
    reward_functions = []
    
    # Add custom reward functions from config
    if config.get('use_length_reward', True):
        reward_functions.append(reward_length)
    
    if config.get('use_capitalization_reward', True):
        reward_functions.append(reward_capitalization)
    
    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_functions,
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    output_dir = config.get('save_dir', f"qwen-0.5b-grpo-{compression}-compression")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
