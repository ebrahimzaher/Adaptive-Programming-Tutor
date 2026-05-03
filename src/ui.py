import gradio as gr
from unsloth import FastLanguageModel
import torch
import sys
from io import StringIO

# --- 1. Load the Model ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen_adaptive_tutor",
    max_seq_length = 256,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# دالة ذكية للتأكد من صحة الكود (Syntax Check)
def check_code_validity(code):
    try:
        compile(code, '<string>', 'exec')
        return True
    except Exception:
        return False

def get_tutor_hint(code_input):
    if not code_input.strip():
        return "Please enter your code first!"
    
    # التأكد من صحة الكود أولاً
    is_valid = check_code_validity(code_input)
    
    try:
        if is_valid:
            # لو الكود صح، هنغير الـ Instruction عشان الموديل يشجعه
            instruction = f"The following code is syntactically correct. Congratulate the student and explain why it's good:\n{code_input}"
        else:
            # لو الكود غلط، نستخدم الـ Instruction القديم بتاع التلميحات
            instruction = f"Analyze this buggy code and provide a Socratic hint to help the student fix it:\n{code_input}"
        
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 128, 
            use_cache = True, 
            temperature = 0.4
        )
        
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        hint = response.split("assistant\n")[-1]
        return hint
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        torch.cuda.empty_cache()

# --- 2. JavaScript for Tab support ---
js_func = """
function() {
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(el => {
        el.addEventListener('keydown', function(e) {
            if (e.key == 'Tab') {
                e.preventDefault();
                var start = this.selectionStart;
                var end = this.selectionEnd;
                this.value = this.value.substring(0, start) + "    " + this.value.substring(end);
                this.selectionStart = this.selectionEnd = start + 4;
            }
        });
    });
}
"""

# --- 3. Gradio UI Layout ---
with gr.Blocks(title="AI Adaptive Programming Tutor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 AI Adaptive Programming Tutor")
    gr.Markdown("Developed by: **Ebrahim Zaher** | Adaptive Learning System")
    
    with gr.Column():
        code_box = gr.Textbox(
            label="Input Your Code", 
            lines=12, 
            placeholder="Paste your code here..."
        )
        submit_btn = gr.Button("Analyze Code ✨", variant="primary")
        
    output_text = gr.Textbox(
        label="Tutor's Feedback", 
        placeholder="Result will appear here...", 
        interactive=False
    )
    
    submit_btn.click(fn=get_tutor_hint, inputs=code_box, outputs=output_text)
    demo.load(None, None, None, js=js_func)

if __name__ == "__main__":
    demo.launch()