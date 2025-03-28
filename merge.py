from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # Thêm thư viện peft
from pathlib import Path
import os

# Đường dẫn mô hình gốc
base_model_path = "unsloth/Qwen2.5-Coder-3B-Instruct"
# base_model_path = "/home/nguyentrung/Documents/git/fineTune_llm/outputs/models/model"
model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Đường dẫn LoRA checkpoint
lora_path = "/home/nguyentrung/Documents/git/fineTune_llm/outputs/loras/lora_model"

# Tải LoRA vào mô hình
model = PeftModel.from_pretrained(model, lora_path)

# Hợp nhất LoRA vào mô hình chính
model = model.merge_and_unload()

# Xác định đường dẫn lưu mô hình sau khi hợp nhất
base_merged_model_dir = "/home/nguyentrung/Documents/git/fineTune_llm/outputs/models"
merged_model_name = "model_final"
merged_model_path = Path(os.path.join(base_merged_model_dir, merged_model_name))
counter = 1

# Kiểm tra xem folder đã tồn tại chưa, nếu có thì tạo folder mới với số tăng dần
while merged_model_path.exists():
    merged_model_path = Path(os.path.join(base_merged_model_dir, f"{merged_model_name}_{counter}"))
    counter += 1

# Tạo thư mục nếu chưa tồn tại
merged_model_path.mkdir(parents=True, exist_ok=True)

# Lưu mô hình và tokenizer vào đường dẫn mới
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Model and tokenizer saved to: {merged_model_path}")
