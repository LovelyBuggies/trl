from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import argparse
import yaml
import os

def parse_arguments():
    """Parse command line arguments for configuration settings."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with configurable settings."
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
    run_name = config.get('run_name', f'qwen-0.5b-{compression}-compression')
    
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
    dataset_name = config.get('dataset', "trl-lib/Capybara")
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
    # For Qwen models, you may need to inspect model architecture to find correct module names
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
    
    # Configure the trainer with parameters from config
    training_args = TrainingArguments(
        output_dir=config.get('output_dir', './results'),
        num_train_epochs=int(config.get('num_train_epochs', 3)),
        per_device_train_batch_size=int(config.get('batch_size', 4)),
        learning_rate=float(config.get('learning_rate', 5e-5)),
        warmup_steps=int(config.get('warmup_steps', 0)),
        logging_dir=config.get('logging_dir', './logs'),
        report_to="wandb" if use_wandb else "none",
        run_name=run_name if use_wandb else None,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    output_dir = config.get('save_dir', f"qwen-0.5b-finetuned-{compression}-compression")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
