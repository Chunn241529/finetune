from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

# Cấu hình chung
max_seq_length = 2048
dtype = None  # Tự động chọn: Float16 cho Tesla T4, Bfloat16 cho Ampere+
load_in_4bit = True

# Bước 1: Tải model gốc đã được huấn luyện
print("Đang tải model gốc...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/nguyentrung/Documents/git/fineTune_llm/outputs/models/model_final",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Bước 2: Áp dụng LoRA mới cho dataset y khoa
print("Đang áp dụng LoRA mới cho dataset y khoa...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Format dataset function for translation dataset as knowledge
def formatting_translation_knowledge_func(examples):
    # Combine source and target into a single text string for knowledge injection
    texts = []
    for i in range(len(examples["en"])):
        # Format as plain text: "Vietnamese: [source] - English: [target]"
        text = f"English: {examples['en'][i]} translate to Vietnamese: {examples['vi'][i]}"
        texts.append(text)
    return {"text": texts}

# Load and format the translator dataset
print("Loading and formatting translator dataset...")
dataset = load_dataset("duwuonline/en_vi_advanced_sentences", split="train")  # Replace with your actual dataset path
dataset = dataset.map(formatting_translation_knowledge_func, batched=True, remove_columns=dataset.column_names)

# Verify formatting (optional)
print("\nExample translation knowledge from dataset:")
print(dataset[0]["text"])


# Bước 4: Thiết lập huấn luyện cho LoRA mới
print("Đang thiết lập trainer cho dataset...")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    dataset_num_proc=2,
    packing=True,  # Có thể đặt True nếu văn bản ngắn để tăng tốc huấn luyện
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=2,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=50,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# Bước 5: Huấn luyện LoRA mới
print("Bắt đầu huấn luyện trên dataset...")
trainer_stats = trainer.train()

# Bước 6: Lưu LoRA mới
print("Đang lưu LoRA mới...")
model.save_pretrained("outputs/loras/LoraTranslatorKnowledge")
tokenizer.save_pretrained("outputs/loras/LoraTranslatorKnowledge")

# In thông tin GPU
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Dung lượng tối đa = {max_memory} GB.")
print(f"{start_gpu_memory} GB bộ nhớ đã được sử dụng.")
