from huggingface_hub import snapshot_download

print("🚀 Starting download for Qwen2.5-1.5B-Instruct...")
print("This will download the model files directly to your hard drive (No VRAM used).")

# تحميل الموديل وحفظه في فولدر محلي
snapshot_download(
    repo_id="unsloth/Qwen2.5-1.5B-Instruct",
    local_dir="./qwen_base_model",       # اسم الفولدر اللي هينزل فيه الموديل
    local_dir_use_symlinks=False,        # عشان ينزل الملفات الحقيقية مش اختصارات
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"] # تجاهل الملفات القديمة، هننزل safetensors بس
)

print("✅ Download Complete! The model is saved in the './qwen_base_model' folder.")