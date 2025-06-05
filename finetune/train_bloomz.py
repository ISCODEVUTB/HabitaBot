"""
train_bloomz.py

Purpose:
    Fine-tunes BLOOMZ-560m on the generated training examples for the HabitaBot project.
    This script should be run from the finetune/ directory.

Dependencies:
    - torch
    - transformers
"""

import torch, gc
import os
torch.cuda.empty_cache(); gc.collect()

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_dataset(file_path, tokenizer, block_size=64):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_file = os.path.join(os.path.dirname(__file__), 'cartagena_rag_finetune.txt')
train_dataset = get_dataset(train_file, tokenizer, block_size=64)
eval_dataset = train_dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir=os.path.join(os.path.dirname(__file__), "bloomz-cartagena-rag-finetuned"),
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_steps=300,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=30,
    report_to="none",
    fp16=False,
    gradient_accumulation_steps=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(os.path.join(os.path.dirname(__file__), "bloomz-cartagena-rag-finetuned"))
tokenizer.save_pretrained(os.path.join(os.path.dirname(__file__), "bloomz-cartagena-rag-finetuned"))
print("¡Fine-tuning RAG completado!")