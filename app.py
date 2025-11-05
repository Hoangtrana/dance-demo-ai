import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import gdown

# Import các module nội bộ
from tutorial_gallery import show_dance_gallery
from pose_utils import extract_multi_person_keypoints, overlay_skeleton_with_scores
from compare_utils_group_avg import compare_dance_group
from ai_feedback_utils import generate_feedback


# =============================
# ⚙️ Streamlit Config
# =============================
st.set_page_config(page_title="Folk Dance Analyzer", layout="wide")
st.title("💃 Folk Dance Analyzer – Ứng dụng học và chấm điểm múa dân gian Việt Nam")
st.caption("Xem bài múa mẫu, luyện tập và nhận phản hồi thông minh 🇻🇳")


# =============================
# 🎥 Google Drive Video Mẫu
# =============================
STANDARD_VIDEO_IDS = {
    "Múa Xòe Tây Bắc": "1Zaj8tGnSgV1Ivtiuk-GImwYGIIu4lUdp",
    "Múa Trống Cơm": "1K4hWlnZk9D_W2T3hQgMhZYcItpzdK8qW"
}


@st.cache_resource
def download_drive_video(drive_id, save_path):
    """Chỉ tải video 1 lần duy nhất."""
    if os.path.exists(save_path):
        return save_path
    url = f"https://drive.google.com/uc?id={drive_id}"
    gdown.download(url, save_path, quiet=False)
    return save_path


# =============================
# 📑 Tabs
# =============================
tab1, tab2 = st.tabs(["🏫 Học Múa", "🧍 Phân tích, So sánh & Chấm Điểm"])


# =============================
# TAB 1 – HỌC MÚA
# =============================
with tab1:
    show_dance_gallery()


# =============================
# TAB 2 – SO SÁNH
# =============================
with tab2:
    st.markdown("### 🎭 Chọn bài múa mẫu để so sánh")

    dance_choice = st.selectbox("🎬 Bài múa:", list(STANDARD_VIDEO_IDS.keys()))
    drive_id = STANDARD_VIDEO_IDS[dance_choice]

    # Đường dẫn lưu cục bộ video mẫu
    os.makedirs("samples/standard", exist_ok=True)
    standard_path = f"samples/standard/{dance_choice.replace(' ', '_')}.mp4"

    # ✅ Tải video mẫu 1 lần duy nhất
    with st.spinner("⏳ Kiểm tra video mẫu..."):
        try:
            standard_path = download_drive_video(drive_id, standard_path)
        except Exception as e:
            standard_path = None
            st.error(f"⚠️ Không tải được video mẫu: {e}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📹 Video mẫu")
        if standard_path and os.path.exists(standard_path):
            st.video(standard_path)
        else:
            st.warning("⚠️ Video mẫu chưa sẵn sàng.")

    with col2:
        uploaded_file = st.file_uploader("📤 Tải video của bạn", type=["mp4", "mov"])

        user_path = None
        if uploaded_file:
            save_dir = f"samples/user_uploads/{dance_choice.replace(' ', '_')}/"
            os.makedirs(save_dir, exist_ok=True)
            user_path = os.path.join(save_dir, uploaded_file.name)
            with open(user_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success("✅ Video đã được tải lên!")


    # =============================
    # 🔍 Chạy phân tích nếu đủ dữ liệu
    # =============================
    if standard_path and user_path:
        st.markdown("---")
        st.subheader("🔍 Phân tích & So sánh chi tiết")

        with st.spinner("🧮 Đang tính điểm tổng thể..."):
            avg_score = compare_dance_group(standard_path, user_path)

        st.success(f"🎯 Điểm trung bình toàn bài: **{avg_score:.1f}/100**")

        st.markdown("### 🦴 Hiển thị khung xương (Pose Skeleton)")
        colA, colB = st.columns(2)

        with st.spinner("🎥 Đang xử lý video khung xương..."):
            standard_overlay = overlay_skeleton_with_scores(standard_path, "temp_standard_pose.mp4", scores=[avg_score])
            user_overlay = overlay_skeleton_with_scores(user_path, "temp_user_pose.mp4", scores=[avg_score])

        with colA:
            st.markdown("**📺 Video mẫu (Pose)**")
            st.video(standard_overlay)

        with colB:
            st.markdown("**🧍 Video của bạn (Pose + Điểm)**")
            st.video(user_overlay)

        st.markdown("### 💬 Gợi ý cải thiện động tác")
        with st.spinner("🧠 Đang tạo phản hồi..."):
            seq_s = extract_multi_person_keypoints(standard_path)
            seq_u = extract_multi_person_keypoints(user_path)

            feedback_list = generate_feedback(
                np.mean(seq_s[0], axis=0) if seq_s else np.zeros(99),
                np.mean(seq_u[0], axis=0) if seq_u else np.zeros(99),
                avg_score
            )

        for fb in feedback_list:
            st.markdown(f"- {fb}")

        st.info("💡 Ứng dụng đang chạy hoàn toàn **Offline** — không cần API & không tốn phí.")
