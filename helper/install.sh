#!/bin/bash

# Kiểm tra xem môi trường ảo đã tồn tại chưa
if [ -d ".venv" ]; then
    echo "Môi trường ảo virtual environment đã tồn tại."
else
    # Tạo mới môi trường ảo
    echo "Đang tạo mới virtual environment..."
    python3 -m venv .venv

    # Kiểm tra việc tạo môi trường ảo có thành công không
    if [ ! -d ".venv" ]; then
        echo "Lỗi khi tạo môi trường ảo virtual environment."
        exit 1
    fi
fi

# Kích hoạt môi trường ảo
echo "Kích hoạt môi trường ảo virtual environment..."
source .venv/bin/activate

echo "Đang cài thư viện..."

# python3 -m pip3 install --upgrade pip
pip3 install torch torchvision torchaudio
pip3 install ipykernel ipywidgets unsloth diffusers pillow
pip3 install git+https://github.com/huggingface/trl.git
pip3 install unsloth "xformers==0.0.28.post2"
pip3 uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip3 install --upgrade transformers
pip3 install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip3 install optimum-quanto
pip3 install python-dotenv

echo "Đã xong!"
