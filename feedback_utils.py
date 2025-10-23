import numpy as np

# Xác định nhóm khớp (MediaPipe Pose có 33 điểm)
JOINT_GROUPS = {
    "tay": [11, 12, 13, 14, 15, 16],
    "chan": [23, 24, 25, 26, 27, 28],
    "than_tren": [11, 12, 23, 24],
}

def analyze_joint_diff(std_seq, user_seq):
    """Tính sai lệch trung bình từng nhóm khớp"""
    diffs = {}
    min_len = min(len(std_seq), len(user_seq))
    if min_len == 0:
        return None

    std_seq = np.array(std_seq[:min_len])
    user_seq = np.array(user_seq[:min_len])
    diff = np.abs(std_seq - user_seq)

    for name, idxs in JOINT_GROUPS.items():
        group_diff = np.mean(diff[:, idxs])
        diffs[name] = group_diff
    return diffs


def generate_feedback(std_seq, user_seq):
    """Sinh phản hồi cụ thể dựa trên sai lệch keypoints"""
    diffs = analyze_joint_diff(std_seq, user_seq)
    if diffs is None:
        return ["Không có dữ liệu để so sánh."]

    feedback = []

    # Đánh giá tổng thể
    avg_diff = np.mean(list(diffs.values()))
    if avg_diff < 0.02:
        feedback.append("✨ Rất tốt! Động tác của bạn gần như khớp hoàn toàn với bài mẫu.")
    elif avg_diff < 0.05:
        feedback.append("👍 Động tác nhìn rất ổn, chỉ cần tinh chỉnh thêm một chút để đạt chuẩn.")
    else:
        feedback.append("👀 Cần chú ý hơn — có một số phần lệch đáng kể so với bài mẫu.")

    # Đánh giá theo từng phần
    if diffs["tay"] > 0.04:
        feedback.append("🖐 Tay: Biên độ vung tay hơi hẹp, thử nâng tay cao và duỗi thẳng hơn.")
    if diffs["chan"] > 0.04:
        feedback.append("🦵 Chân: Nhịp di chuyển chưa đều, tập giữ thăng bằng và nhịp đều.")
    if diffs["than_tren"] > 0.04:
        feedback.append("💃 Thân trên: Tư thế hơi nghiêng, cần giữ vai và eo thẳng hơn.")

    if len(feedback) == 1:
        feedback.append("Tuyệt vời! Bạn đã thực hiện đúng toàn bộ động tác 🎉")

    return feedback
