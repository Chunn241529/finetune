@echo off

REM Kiểm tra xem môi trường ảo đã tồn tại chưa
IF EXIST .venv (
    echo Môi trường ảo virtual environment đã tồn tại.
) ELSE (
    REM Tạo mới môi trường ảo
    echo Đang tạo mới virtual environment...
    python -m venv .venv

    REM Kiểm tra việc tạo môi trường ảo có thành công không
    IF NOT EXIST .venv (
        echo Lỗi khi tạo môi trường ảo virtual environment.
        exit /b 1
    )
)

REM Kích hoạt môi trường ảo
echo Kích hoạt môi trường ảo virtual environment...
call .venv\Scripts\activate

python.exe -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install ipykernel
pip install vllm
pip install --upgrade diffusers[torch]


@REM REM Kiểm tra và xóa tệp requirements.txt nếu tồn tại
@REM IF EXIST requirements.txt (
@REM     echo Đang xóa tệp requirements.txt...
@REM     del /q requirements.txt
@REM )

@REM REM Tạo tệp requirements.txt mới với các thư viện cần thiết
@REM echo Khởi tạo file requirements.txt...
@REM (
@REM     echo openai
@REM     echo fastapi
@REM     echo uvicorn
@REM     echo python-dotenv
@REM     echo duckduckgo-search
@REM     echo requests
@REM     echo numpy
@REM     echo pillow
@REM     echo protobuf
@REM     echo tqdm
@REM     echo gfpgan
@REM     echo schedule
@REM     echo pygments
@REM     echo beautifulsoup4
@REM     echo python-multipart
@REM     echo PyJWT
@REM     echo httpx
@REM     echo aiohttp
@REM     echo passlib
@REM     echo argon2_cffi
@REM     echo torch
@REM     echo transformers
@REM     echo datasets
@REM     echo accelerate
@REM     echo pywin32
@REM     echo huggingface_hub
@REM ) > requirements.txt



@REM REM Cài đặt và cập nhật các gói từ requirements.txt
@REM echo Tiến hành tải packages trong requirements.txt...
@REM pip install -U -r requirements.txt || echo Lỗi khi tải packages

echo Đã xong!
pause
