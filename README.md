# 🎓 Adaptive Programming Tutor (Fine-Tuned LLM Application)

[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/ebrahimzaher/qwen_adaptive_tutor)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)

An end-to-end machine learning project that fine-tunes an open-source LLM to act as a **Socratic Programming Instructor**. Instead of giving direct code solutions, the model analyzes buggy code and provides conceptual, encouraging hints to help students learn how to debug independently.

## 🚀 Features

- **Synthetic Data Generation Pipeline:** Built a robust data generation script using `LangChain` and `Llama-3/4` (via Groq API) to create a custom dataset of buggy code and Socratic hints.
- **Efficient Fine-Tuning (PEFT):** Fine-tuned the `Qwen2.5-1.5B-Instruct` model locally using **Unsloth** and **LoRA** (Low-Rank Adaptation).
- **Smart Inference UI:** Developed an interactive **Gradio** web interface featuring pre-inference syntax validation.
- **Secure Configuration:** Utilizes environment variables (`.env`) for safe API Key and Access Token management.

## 🛠️ Tech Stack

- **Data Generation:** LangChain, Groq API, Pandas.
- **Model & Fine-Tuning:** PyTorch, Unsloth, Hugging Face (Transformers, PEFT, Datasets), LoRA.
- **Deployment & UI:** Gradio, Python-dotenv.

## 📁 Project Structure

```text
├── data/
│   ├── unlabeled_data.csv       # Raw buggy/fixed code pairs
│   └── labeled_data.csv         # Generated dataset with Socratic hints
├── src/
│   ├── data_labeling.py         # Script for synthetic data generation via Groq
│   ├── train.py                 # Fine-tuning script using Unsloth & LoRA
│   ├── upload_to_hf.py          # Script to push the model to Hugging Face
│   └── app.py                   # Gradio UI and Inference logic
├── requirements.txt             # Project dependencies
└── README.md