#!/bin/bash

# Script để tải và cài đặt NVIDIA Driver và CUDA 12.4 trên Linux (Ubuntu)

# Kiểm tra quyền root
if [ "$EUID" -ne 0 ]; then
  echo "Vui lòng chạy script này với quyền sudo!"
  exit 1
fi

# Cập nhật hệ thống và cài đặt các gói cần thiết
echo "Đang cập nhật hệ thống và cài đặt các gói cần thiết..."
apt-get update -y
apt-get install -y build-essential gcc g++ make dkms

# Tạo thư mục tạm để tải file
TEMP_DIR="/tmp/nvidia_install"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# URL tải CUDA 12.4 runfile (cho Ubuntu 22.04, x86_64)
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
CUDA_FILE="cuda_12.4.0_550.54.14_linux.run"

# Tải file cài đặt CUDA 12.4
echo "Đang tải CUDA 12.4 từ NVIDIA..."
wget -O "$CUDA_FILE" "$CUDA_URL"

# Kiểm tra xem tải thành công chưa
if [ ! -f "$CUDA_FILE" ]; then
  echo "Lỗi: Không thể tải file CUDA. Kiểm tra kết nối mạng hoặc URL."
  exit 1
fi

# Cấp quyền thực thi cho file run
chmod +x "$CUDA_FILE"

# Cài đặt driver NVIDIA và CUDA Toolkit
echo "Đang cài đặt NVIDIA Driver và CUDA 12.4..."
./"$CUDA_FILE" --silent --driver --toolkit

# Kiểm tra kết quả cài đặt
if [ $? -eq 0 ]; then
  echo "Cài đặt thành công!"
else
  echo "Lỗi trong quá trình cài đặt. Xem log tại /var/log/nvidia-installer.log hoặc /var/log/cuda-installer.log"
  exit 1
fi

# Thiết lập biến môi trường cho CUDA
echo "Thiết lập biến môi trường cho CUDA..."
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

# Xóa thư mục tạm
echo "Dọn dẹp file tạm..."
rm -rf "$TEMP_DIR"

# Kiểm tra cài đặt
echo "Kiểm tra driver NVIDIA:"
nvidia-smi
echo "Kiểm tra CUDA:"
nvcc --version

echo "Hoàn tất! Vui lòng khởi động lại hệ thống để áp dụng thay đổi."
