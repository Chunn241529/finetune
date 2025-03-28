# Finetune - Tinh Chỉnh Mô Hình Ngôn Ngữ Lớn với Unsloth

[![GitHub stars](https://img.shields.io/github/stars/Chunn241529/finetune?style=social)](https://github.com/Chunn241529/finetune/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Chunn241529/finetune?style=social)](https://github.com/Chunn241529/finetune/network)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Chunn241529/finetune/blob/main/LICENSE)

**Finetune** là một dự án mẫu để tinh chỉnh (finetuning) các mô hình ngôn ngữ lớn (LLM) như `Qwen2.5`, `Llama-3.1`, `Mistral`, và nhiều mô hình khác từ Hugging Face, sử dụng thư viện `unsloth`. Dự án được thiết kế để tối ưu hiệu suất, giảm sử dụng bộ nhớ với hỗ trợ quantization 4-bit, và dễ dàng triển khai trên các hệ thống có GPU NVIDIA.

Dù bạn là nhà nghiên cứu, lập trình viên hay người đam mê AI, dự án này sẽ giúp bạn tùy chỉnh LLM cho các tác vụ cụ thể như tạo văn bản, dịch thuật, hoặc lập trình tự động!

---

## ✨ Tính Năng Nổi Bật
- **Hiệu suất cao**: Tăng tốc độ tải mô hình và giảm bộ nhớ với `unsloth` (4-bit quantization).
- **Dễ triển khai**: Chỉ vài bước cài đặt là có thể chạy.
- **Bảo mật tốt**: Lưu trữ token nhạy cảm trong `.env`, tránh lộ thông tin.
- **Hỗ trợ đa dạng**: Tinh chỉnh nhiều mô hình từ Hugging Face như `Qwen2.5-Coder`, `Llama-3.1`, `Gemma`, v.v.
- **Tài liệu chi tiết**: Hướng dẫn từng bước kèm script tự động.

## 🛠 Yêu Cầu Hệ Thống
- **Hệ điều hành**: Linux (đã thử nghiệm trên Ubuntu 22.04).
- **GPU**: NVIDIA GPU hỗ trợ CUDA (khuyến nghị ≥ 16GB VRAM cho mô hình lớn).
- **Phần mềm**: 
  - Python 3.8+
  - Git
  - NVIDIA Driver (550.54.14+)
  - CUDA Toolkit 12.4

---

## 📥 Hướng Dẫn Cài Đặt

### 1. Chuẩn Bị Môi Trường

#### Cài đặt NVIDIA Driver và CUDA 12.4
  1. Tải script cài đặt tự động:
      ```bash
      wget https://raw.githubusercontent.com/Chunn241529/finetune/main/install_nvidia_cuda.sh
      ```
  2. Cấp quyền và chạy:
      ```bash
      chmod +x install_nvidia_cuda.sh
      sudo ./install_nvidia_cuda.sh
      ```
  3. Khởi động lại hệ thống:
      ```bash
      sudo reboot
      ```
  4. Kiểm tra cài đặt:
      ```bash
      nvidia-smi  # Xem thông tin driver và GPU
      nvcc --version  # Xác nhận CUDA 12.4
      ```
  5. Cài đặt python và git
      ```bash
      sudo apt update
      sudo apt install -y python3 python3-pip git
      ```
  6. Git clone source:
      ```bash
      git clone https://github.com/Chunn241529/finetune.git
      cd finetune
      ```
  7. Cài đặt VSCode và các extensions liên quan:
    1. Các extensions python.
    2. Các extensions Jupyter.
  8. Cài đặt thư viện:
      ```bash
      cd helper
      ./install.sh
      ```
  9. Thiết Lập Token Hugging Face:
    1. Tạo tệp .env để lưu token:
      ```bash
      echo "HF_TOKEN=your_huggingface_token_here" > .env 
      ```
      - Lấy token tại [Hugging Face Settings.](https://huggingface.co/settings/tokens)
      - Thay `your_huggingface_token_here` bằng token của bạn.
    2. Đảm bảo `.env` được bỏ qua trong `.gitignore` (đã có sẵn).


## 🚀 Cách Sử Dụng
1. Chỉnh sửa `finetune.ipynb` nếu cần (ví dụ: thay đổi model_name).
2. Bấm nút run all
