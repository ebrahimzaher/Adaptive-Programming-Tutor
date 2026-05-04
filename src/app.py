from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import evaluate
from tqdm import tqdm

if __name__ == "__main__":

    print("Loading Dataset...")
    df = pd.read_csv("data/labeled_data.csv").dropna()
    df = df.drop_duplicates()
    dataset = Dataset.from_pandas(df)
    
    split_dataset = dataset.train_test_split(test_size=0.1, seed=3407)


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

    print("Tokenizing Datasets...")
    def tokenize_func(example):
        text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return tokenizer(text, truncation=True, max_length=512)

    train_dataset = split_dataset["train"].map(tokenize_func, remove_columns=dataset.column_names)
    eval_dataset = split_dataset["test"].map(tokenize_func, remove_columns=dataset.column_names)


    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
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


    print("\n" + "="*50)
    print("📊 Starting ROUGE Evaluation on Test Set...")
    
    rouge_metric = evaluate.load("rouge")
    FastLanguageModel.for_inference(model)
    
    test_sample = split_dataset["test"]
    
    test_size_to_evaluate = min(50, len(test_sample))
    test_sample = test_sample.select(range(test_size_to_evaluate)) 
    
    predictions = []
    references = test_sample["output"]
    
    for instruction in tqdm(test_sample["instruction"], desc="Generating Responses"):
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=0.3)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        hint = response.split("assistant\n")[-1].strip()
        predictions.append(hint)
    
    results = rouge_metric.compute(predictions=predictions, references=references)
    
    print("\n📈 ROUGE Scores:")
    print(f"ROUGE-1 (Word Match)       : {results['rouge1']:.4f}")
    print(f"ROUGE-2 (Bigram Match)     : {results['rouge2']:.4f}")
    print(f"ROUGE-L (Longest Sequence) : {results['rougeL']:.4f}")
    print("="*50)
    print("🚀 Full Pipeline Executed Successfully!")