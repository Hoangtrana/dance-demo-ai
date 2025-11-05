import cv2
import numpy as np
from ultralytics import YOLO

# ================================
# 🎯 CẤU HÌNH MODEL YOLOv8-Pose
# ================================
# model nhẹ, phù hợp demo hoặc CPU
YOLO_MODEL = YOLO("yolov8n-pose.pt")

# ================================
# 🧍‍♀️ Trích xuất keypoints nhiều người
# ================================
def extract_multi_person_keypoints(video_path, max_people=5):
    """
    Trích xuất pose keypoints từ video có nhiều người múa.
    Trả về danh sách mảng numpy [person_1, person_2, ...]
    """
    cap = cv2.VideoCapture(video_path)
    people_sequences = [[] for _ in range(max_people)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = YOLO_MODEL(frame, verbose=False)
        if len(results) == 0:
            continue

        keypoints_all = results[0].keypoints
        if keypoints_all is None:
            continue

        poses = keypoints_all.xy.cpu().numpy()  # (N, 17, 2)
        n_people = min(len(poses), max_people)

        for i in range(n_people):
            coords = poses[i].flatten()
            people_sequences[i].append(coords)

    cap.release()
    # chỉ trả về những người có dữ liệu
    return [np.array(seq) for seq in people_sequences if len(seq) > 0]


# ================================
# 🧮 Trung bình khung xương nhóm
# ================================
def average_group_pose(people_sequences):
    """Tính trung bình khung xương của nhóm."""
    if not people_sequences:
        return np.zeros((1, 34))  # 17 điểm * 2 tọa độ
    min_len = min(len(seq) for seq in people_sequences)
    trimmed = [seq[:min_len] for seq in people_sequences]
    return np.mean(trimmed, axis=0)


# ================================
# 🎥 Hiển thị skeleton + điểm từng người
# ================================
def overlay_skeleton_with_scores(video_path, output_path="temp_overlay.mp4", scores=None):
    """
    Hiển thị khung xương (pose skeleton) và điểm từng người trên video.
    Hỗ trợ multi-person từ YOLOv8-Pose.
    """
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    # ✅ Danh sách kết nối khớp (17 keypoints theo YOLOv8)
    # Mỗi tuple là cặp chỉ số hai điểm cần nối
    SKELETON_CONNECTIONS = [
        (5, 7), (7, 9),    # tay phải
        (6, 8), (8, 10),   # tay trái
        (5, 6),            # vai nối nhau
        (11, 12),          # hông nối nhau
        (5, 11), (6, 12),  # thân
        (11, 13), (13, 15),  # chân phải
        (12, 14), (14, 16)   # chân trái
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = YOLO_MODEL(frame, verbose=False)
        frame_vis = frame.copy()

        if len(results) > 0 and results[0].keypoints is not None:
            poses = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
            for i, pts in enumerate(poses):
                pts = pts.astype(int)
                color = COLORS[i % len(COLORS)]

                # Vẽ đường nối giữa các khớp (connections)
                for a, b in SKELETON_CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        xa, ya = pts[a]
                        xb, yb = pts[b]
                        cv2.line(frame_vis, (xa, ya), (xb, yb), color, 2)

                # Vẽ điểm khớp
                for (x, y) in pts:
                    cv2.circle(frame_vis, (x, y), 3, color, -1)

                # Tính trung tâm để hiển thị nhãn
                x_mean, y_mean = np.mean(pts, axis=0).astype(int)

                if scores and i < len(scores):
                    label = f"P {i+1}: {scores[i]:.1f}"
                else:
                    label = f"P {i+1}"

                cv2.putText(frame_vis, label, (x_mean - 40, y_mean - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        out.write(frame_vis)

    cap.release()
    out.release()
    return output_path

