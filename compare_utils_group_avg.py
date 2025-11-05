import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist

from pose_utils import extract_multi_person_keypoints, average_group_pose

def dynamic_time_warping(seqA, seqB):
    """DTW khoảng cách giữa hai chuỗi pose"""
    lenA, lenB = len(seqA), len(seqB)
    cost = np.full((lenA + 1, lenB + 1), np.inf)
    cost[0, 0] = 0

    for i in range(1, lenA + 1):
        for j in range(1, lenB + 1):
            dist = np.linalg.norm(seqA[i - 1] - seqB[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[lenA, lenB] / (lenA + lenB)


def compare_dance_group(std_video, usr_video):
    """Chấm điểm nhóm dựa trên trung bình toàn bộ người múa"""
    std_people = extract_multi_person_keypoints(std_video)
    usr_people = extract_multi_person_keypoints(usr_video)

    # Nếu không có keypoints
    if not std_people or not usr_people:
        return 0.0

    seq_standard = average_group_pose(std_people)
    seq_user = average_group_pose(usr_people)

    # Đồng bộ độ dài
    min_len = min(len(seq_standard), len(seq_user))
    seq_standard, seq_user = seq_standard[:min_len], seq_user[:min_len]

    # So sánh tương đồng
    try:
        diff = np.mean([cosine(a, b) for a, b in zip(seq_standard, seq_user)])
    except:
        diff = dynamic_time_warping(seq_standard, seq_user) / 100

    base_score = max(0, 100 - diff * 100)

    # Tính độ đồng bộ nhóm (phương sai)
    var_std = np.mean([np.var(p) for p in std_people])
    var_usr = np.mean([np.var(p) for p in usr_people])
    sync_factor = max(0.8, 1 - abs(var_std - var_usr) * 5)

    final_score = round(base_score * sync_factor, 1)
    return final_score
