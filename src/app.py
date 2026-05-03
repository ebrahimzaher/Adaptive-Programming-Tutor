from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

if __name__ == "__main__":
    
    print("Loading Qwen 1.5B Model from local folder...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./qwen_base_model", 
        max_seq_length = 512,   
        dtype = None, 
        load_in_4bit = True, 
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("Loading Dataset...")
    df = pd.read_csv("data/labeled_data.csv")
    dataset = Dataset.from_pandas(df)

    def tokenize_func(example):
        text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return tokenizer(text, truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_func, remove_columns=dataset.column_names)
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=3407)

    trainer = Trainer(
        model = model,
        train_dataset = split_dataset["train"],
        eval_dataset = split_dataset["test"],
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args = TrainingArguments(
            per_device_train_batch_size = 1, 
            gradient_accumulation_steps = 8, 
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            eval_strategy = "steps",
            eval_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            output_dir = "outputs",
            gradient_checkpointing = True, 
        ),
    )

    print("Starting Training...")
    trainer.train()

    print("Saving Model...")
    model.save_pretrained("qwen_adaptive_tutor")
    tokenizer.save_pretrained("qwen_adaptive_tutor")
    
    print("🎉 Training Complete! Your model is saved in the 'qwen_adaptive_tutor' folder.")
