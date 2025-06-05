"""
generate_training_txt.py

Purpose:
    Converts .jsonl prompt-completion examples to .txt format for language model fine-tuning.
    This script should be run from the finetune/ directory.

Dependencies:
    - json
"""

import json
import os

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), 'cartagena_rag_finetune.jsonl')
    output_path = os.path.join(os.path.dirname(__file__), 'cartagena_rag_finetune.txt')
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            obj = json.loads(line)
            fout.write(obj["prompt"].strip() + "\n" + obj["completion"].strip() + "\n\n")