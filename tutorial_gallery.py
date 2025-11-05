import streamlit as st
import os
import base64

# =============================
# 📚 DỮ LIỆU CÁC BÀI MÚA
# =============================
tutorials = [
    {
        "name": "Múa Xòe Tây Bắc",
        "desc": "Múa xòe là một loại hình nghệ thuật dân gian đặc sắc của đồng bào Thái ở vùng Tây Bắc Việt Nam.",
        "thumb": "samples/tutorials/xoe_thumb.jpg",
        "drive_id": "1cc9nrqU0sdi7_27os-4z60UTbjN3W2FO",
    },
    {
        "name": "Học múa Xòe",
        "desc": "Bài múa dân gian Tây Bắc – nhịp 3/4, động tác nối vòng thể hiện tinh thần đoàn kết.",
        "thumb": "samples/tutorials/xoe_thumb_learn.jpg",
        "drive_id": "1Zaj8tGnSgV1Ivtiuk-GImwYGIIu4lUdp",
    },
    {
        "name": "Múa Trống Cơm",
        "desc": "Điệu múa thể hiện niềm vui, sự hứng khởi, và khát vọng về cuộc sống no đủ, hạnh phúc.",
        "thumb": "samples/tutorials/trongcom_thumb.jpg",
        "drive_id": "1K4hWlnZk9D_W2T3hQgMhZYcItpzdK8qW",
    },
    {
        "name": "Học múa Trống Cơm",
        "desc": "Đặc trưng miền Bắc, tiết tấu rộn ràng, động tác phối hợp tay – hông linh hoạt.",
        "thumb": "samples/tutorials/trongcom_thumb_learn.jpg",
        "drive_id": "1ZKBhqwnCAW1EXPw0MmfM9Vl5pD0p-l69",
    }
]

# =============================
# 🖼️ Hàm load ảnh thumb an toàn
# =============================
def load_thumbnail(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        ext = path.split(".")[-1]
        b64 = base64.b64encode(data).decode()
        return f"data:image/{ext};base64,{b64}"
    else:
        return "https://placehold.co/400x300?text=No+Image"

# =============================
# 🎨 GIAO DIỆN HIỂN THỊ
# =============================
def show_dance_gallery():
    st.markdown("## 🏫 Học Múa")
    st.caption("Chọn một bài múa để xem video hướng dẫn 💃")

    cols = st.columns(2)

    for i, t in enumerate(tutorials):
        with cols[i % 2]:
            img_src = load_thumbnail(t["thumb"])
            st.markdown(
                f'<img src="{img_src}" style="width:100%;height:350px;object-fit:cover;border-radius:8px;">',
                unsafe_allow_html=True,
            )
            st.markdown(f"### {t['name']}")
            st.caption(t["desc"])

            if st.button(f"🎥 Xem video - {t['name']}", key=f"btn_{i}"):
                st.session_state["selected_tutorial"] = t

    # Khi người dùng chọn video
    if "selected_tutorial" in st.session_state:
        tutorial = st.session_state["selected_tutorial"]
        st.markdown("---")
        st.markdown(f"## 🎬 {tutorial['name']}")

        iframe_html = f"""
        <iframe src="https://drive.google.com/file/d/{tutorial['drive_id']}/preview"
        width="100%" height="480" allow="autoplay"></iframe>
        """
        st.markdown(iframe_html, unsafe_allow_html=True)

        st.caption("💡 Bạn có thể xem toàn màn hình hoặc pause video để quan sát động tác.")
