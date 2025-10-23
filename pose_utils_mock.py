import numpy as np
import tempfile

def extract_keypoints_from_video(video_path):
    # Trả dữ liệu giả để mô phỏng pose (33 điểm, 3 tọa độ)
    frames = []
    for _ in range(50):
        frames.append(np.random.rand(33, 3))
    return frames

def overlay_skeleton(video_path, output_path):
    # Không xử lý thực, chỉ trả lại video gốc
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(b"")  # placeholder rỗng
    return video_path
