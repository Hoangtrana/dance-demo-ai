import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from pose_utils import extract_keypoints_from_video, overlay_skeleton
from compare_utils import compare_dances, frame_similarity
from feedback_utils import generate_feedback

st.set_page_config(page_title="Dance Pose Analyzer", layout="wide")

st.title("💃 Ứng dụng chấm điểm & so sánh động tác múa")
st.caption("Chọn bài múa chuẩn, tải video của bạn và xem so sánh trực quan 🇻🇳")

# 1️⃣ Chọn bài múa chuẩn
dance_options = {
    "Múa xoè": "samples/xoè/standard.mp4",
    "Múa quạt": "samples/quat/standard.mp4",
    "Múa nón": "samples/non/standard.mp4"
}

dance_choice = st.selectbox("🎭 Chọn bài múa:", list(dance_options.keys()))
standard_path = dance_options[dance_choice]

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📹 Video mẫu")
    if os.path.exists(standard_path):
        st.video(standard_path)
    else:
        st.warning(f"⚠️ Thiếu video mẫu: {standard_path}")

# 2️⃣ Upload video người học
uploaded_file = st.file_uploader("📤 Tải video của bạn", type=["mp4", "mov"])

user_path = None
if uploaded_file:
    save_dir = f"samples/user_uploads/{dance_choice.replace(' ', '_')}/"
    os.makedirs(save_dir, exist_ok=True)
    user_path = os.path.join(save_dir, uploaded_file.name)
    with open(user_path, "wb") as f:
        f.write(uploaded_file.read())
    with col2:
        st.markdown("### 🎥 Video của bạn")
        st.video(user_path)

# 3️⃣ Khi có cả 2 video
if user_path and os.path.exists(standard_path):
    st.markdown("---")
    st.subheader("🔍 So sánh chi tiết")

    # Hiển thị song song video skeleton
    st.markdown("### 🦴 Hiển thị khung xương (Pose Skeleton)")
    colA, colB = st.columns(2)
    with st.spinner("Đang xử lý skeleton..."):
        standard_overlay = overlay_skeleton(standard_path, "temp_standard.mp4")
        user_overlay = overlay_skeleton(user_path, "temp_user.mp4")

    with colA:
        st.markdown("**📺 Video mẫu (pose)**")
        st.video(standard_overlay)
    with colB:
        st.markdown("**🧍 Video của bạn (pose)**")
        st.video(user_overlay)

    # Tính điểm theo thời gian
    st.markdown("### 📈 Biểu đồ khớp động tác theo thời gian")

    seq_standard = extract_keypoints_from_video(standard_path)
    seq_user = extract_keypoints_from_video(user_path)
    frame_scores = frame_similarity(seq_standard, seq_user)
    avg_score = compare_dances(seq_standard, seq_user)

    fig, ax = plt.subplots()
    ax.plot(frame_scores, label="Điểm từng frame")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Thời gian (frame)")
    ax.set_ylabel("Điểm khớp (%)")
    ax.set_title(f"Độ khớp động tác - {dance_choice}")
    ax.legend()
    st.pyplot(fig)

    # Thanh trượt đồng bộ video
    st.markdown("### 🎚️ Tua video đồng bộ")
    total_frames = min(len(seq_standard), len(seq_user))
    frame_idx = st.slider("Chọn vị trí (frame)", 0, total_frames - 1, 0)

    st.info(f"📍 Đang xem frame thứ {frame_idx} – điểm: {frame_scores[frame_idx]:.1f}%")

    st.success(f"🎯 Điểm trung bình toàn bài: **{avg_score}/100**")
        # 🧠 GỢI Ý CẢI THIỆN CỤ THỂ
    st.markdown("### 🧠 Gợi ý cải thiện động tác")

    feedback_list = generate_feedback(seq_standard, seq_user)
    for fb in feedback_list:
        st.markdown(f"- {fb}")


