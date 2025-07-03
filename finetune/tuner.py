from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import torch


# Load tokenizer and model
model_id = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
).to("cpu")

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_dataset("json", data_files="BlendNet.json")

# Preprocess function
def preprocess(example):
    text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['script']}<|im_end|>"
    return tokenizer(text, truncation=True, padding="max_length", max_length=1024)

# Apply preprocessing
tokenized_dataset = dataset["train"].map(
    preprocess,
    remove_columns=dataset["train"].column_names
)

# Custom collator that sets labels
class DataCollatorForCausalLMWithLabels:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad = DataCollatorWithPadding(tokenizer, return_tensors="pt", padding=True)

    def __call__(self, features):
        batch = self.pad(features)
        batch["labels"] = batch["input_ids"].clone()
        return batch

data_collator = DataCollatorForCausalLMWithLabels(tokenizer)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./qwen3-blendnet",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    report_to="none",
    use_cpu=True 
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()
