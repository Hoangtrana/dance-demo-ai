import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compare_dances(seq_standard, seq_user):
    """So sánh toàn bài và tính điểm trung bình"""
    if len(seq_standard) == 0 or len(seq_user) == 0:
        return 0.0

    distance, path = fastdtw(seq_standard, seq_user, dist=euclidean)
    normalized = np.clip(100 - distance / 50, 0, 100)
    return round(float(normalized), 2)

def frame_similarity(seq_standard, seq_user):
    """Tính độ khớp từng frame (timeline)"""
    min_len = min(len(seq_standard), len(seq_user))
    if min_len == 0:
        return []

    frame_scores = []
    for i in range(min_len):
        diff = euclidean(seq_standard[i], seq_user[i])
        score = np.clip(100 - diff * 10, 0, 100)
        frame_scores.append(score)

    return frame_scores
