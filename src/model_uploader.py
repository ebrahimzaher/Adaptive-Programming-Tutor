import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API_KEY")

def load_tutor_model():
    """
    Function to load the base Qwen model and apply the fine-tuned LoRA adapter.
    """
    print("⏳ Loading tokenizer and base model... (This might take a moment if not cached)")
    
    base_model_name = "unsloth/Qwen2.5-1.5B-Instruct"
    adapter_repo = "ebrahimzaher/qwen_adaptive_tutor"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            token=hf_token
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )

        print("🔌 Applying LoRA adapter from Hugging Face...")
        model = PeftModel.from_pretrained(
            base_model, 
            adapter_repo, 
            token=hf_token
        )

        print("✅ Model and Adapter loaded successfully!")
        return tokenizer, model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

if __name__ == "__main__":
    tokenizer, model = load_tutor_model()
    if model:
        print("Ready for inference!")
