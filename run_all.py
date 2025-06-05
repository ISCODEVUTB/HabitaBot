"""
run_all.py

Purpose:
    Master script to run the entire HabitaBot pipeline from start to finish.

Runs:
    1. Data processing.
    2. Prompt generation and training file creation.
    3. BLOOMZ model fine-tuning.
    4. Inference pipeline.
    5. Evaluation and visualization.

Note:
    Make sure the data file is in data/ and dependencies are installed.
"""

import os
import subprocess
import sys

def run_script(path):
    print(f"\n--- Running: {path} ---\n")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

if __name__ == "__main__":
    # 1. Data processing
    run_script(os.path.join('processing', 'data_processing.py'))

    # 2. Prompt generation (jsonl)
    run_script(os.path.join('processing', 'prompt_generation.py'))

    # 3. Convert to .txt for training
    os.chdir('finetune')
    run_script('generate_training_txt.py')
    
    # 4. Fine-tune BLOOMZ model
    run_script('train_bloomz.py')

    # 5. Inference pipeline
    run_script('inference_pipeline.py')

    os.chdir('..')

    # 6. Evaluation and visualization
    run_script(os.path.join('evaluation', 'metrics.py'))
    run_script(os.path.join('evaluation', 'visualization.py'))