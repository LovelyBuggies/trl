from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import MAGRPOConfig, MAGRPOTrainer

from .rewards import reward_length, reward_capitalization
from .utils import parse_arguments, load_config

def main():
    args = parse_arguments()

    """ Set up config """
    config = {}
    if args.config:
        config = load_config(args.config)

    use_wandb = config.get('use_wandb', False)
    wandb_project = config.get('wandb_project', 'trl')
    wandb_entity = config.get('wandb_entity', None)
    run_name = config.get('run_name', None)
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "model_name": config.get('model_name', "Qwen/Qwen2.5-0.5B"),
                "learning_rate": config.get('learning_rate', 5e-5),
                "epochs": config.get('num_train_epochs', 3),
                "batch_size": config.get('batch_size', 4),
            }
        )

    """ Dataset configuration """
    dataset_name = config.get('dataset', "trl-lib/tldr")
    dataset_split = config.get('dataset_split', "train")
    dataset = load_dataset(dataset_name, split=dataset_split)

    """ Model configuration """
    model_name = config.get('model_name', "Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    
    """ LoRA configuration """
    lora_r = config.get('lora_r', 32)
    lora_alpha = config.get('lora_alpha', 64)
    target_modules = config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        fan_in_fan_out=False,
        modules_to_save=None,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    """ Training configuration """
    training_args = MAGRPOConfig(
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
        max_prompt_length=config.get('max_prompt_length', 512),
        max_completion_length=config.get('max_completion_length', 128),
        num_generations=config.get('num_generations', 8),
        temperature=config.get('temperature', 1.0),
        top_p=config.get('top_p', 0.9),
        reward_weights=config.get('reward_weights', None),
        report_to="wandb" if use_wandb else "none",
        run_name=run_name if use_wandb else None,
    )
    reward_functions = [reward_length, reward_capitalization]
    
    """ Training the model """
    trainer = MAGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_functions,
    )
    trainer.train()
    
    """ Save the model """
    output_dir = config.get('save_dir', f"qwen-0.5b-magrpo-tldr")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
