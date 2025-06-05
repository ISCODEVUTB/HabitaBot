"""
inference_pipeline.py

Purpose:
    Loads the fine-tuned model and enables inference using the Hugging Face pipeline.
    This script should be run from the finetune/ directory.

Dependencies:
    - transformers
"""

from transformers import pipeline
import os

finetuned_path = os.path.join(os.path.dirname(__file__), "bloomz-cartagena-rag-finetuned")
pipe = pipeline("text-generation", model=finetuned_path, tokenizer=finetuned_path, max_new_tokens=60, device=0)