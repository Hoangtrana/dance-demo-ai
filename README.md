# Dance Pose Analyzer 💃

Ứng dụng web chấm điểm động tác múa truyền thống Việt Nam bằng AI.
Triển khai bằng Streamlit.

# 1. Clone project

git clone https://github.com/Hoangtrana/dance-demo-ai.git
cd dance-demo-ai

# 2. Tạo môi trường ảo

python3 -m venv venv
source venv/bin/activate

# 3. Cập nhật pip

pip install --upgrade pip

# 4. Cài numpy + OpenCV cho Mac M1/M2/M3

pip uninstall -y numpy opencv-python opencv-python-headless opencv-contrib-python
pip install numpy==1.23.5
pip install opencv-python-headless==4.5.5.64
brew install opencv

# 5. Cài YOLO & PyTorch (CPU)

pip install ultralytics==8.1.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Cài dotenv (nếu dùng AI feedback)

pip install python-dotenv

## Chạy thử local

source venv/bin/activate

```bash
streamlit run app.py
```
