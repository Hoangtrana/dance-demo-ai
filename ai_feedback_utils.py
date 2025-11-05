import os
import numpy as np
from dotenv import load_dotenv

# Load key từ file .env nếu có
load_dotenv()

# ===========================================================
# 1️⃣ Kiểm tra xem có API key OpenAI hay Gemini hay không
# ===========================================================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

openai_client = None
genai_model = None

if OPENAI_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_KEY)
        print("✅ OpenAI client initialized.")
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo OpenAI client: {e}")

elif GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        genai_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini model initialized.")
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo Gemini client: {e}")
else:
    print("💡 Không có API key. Sử dụng chế độ offline (rule-based).")


# ===========================================================
# 2️⃣ Hàm phụ trợ: nội suy khớp độ dài giữa 2 chuỗi pose
# ===========================================================
def _resample_sequence(seq, target_len):
    """Nội suy tuyến tính để khớp độ dài 2 chuỗi pose"""
    if len(seq) == target_len:
        return seq
    if len(seq) == 0:
        return np.zeros((target_len, seq.shape[1]))  # tránh lỗi
    idx_old = np.linspace(0, 1, len(seq))
    idx_new = np.linspace(0, 1, target_len)
    return np.array([np.interp(idx_new, idx_old, seq[:, i]) for i in range(seq.shape[1])]).T


# ===========================================================
# 3️⃣ Hàm tạo phản hồi AI thực (OpenAI / Gemini)
# ===========================================================
def _generate_openai_feedback(mean_diff, motion_var, avg_score):
    prompt = f"""
    Bạn là huấn luyện viên múa Việt Nam.
    Hãy đánh giá bài múa dựa trên thông tin sau:
    - Độ lệch tư thế trung bình: {mean_diff:.3f}
    - Độ mượt chuyển động: {motion_var:.3f}
    - Điểm trung bình: {avg_score:.1f}/100

    Viết 3–4 gợi ý ngắn gọn, thân thiện bằng tiếng Việt:
    - Nhận xét tổng thể (giống hay khác mẫu)
    - Gợi ý cải thiện động tác tay/chân
    - Gợi ý về nhịp và cảm xúc
    - Câu động viên cuối
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là huấn luyện viên múa Việt Nam, nói ngắn gọn, khích lệ."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return [response.choices[0].message.content]
    except Exception as e:
        print(f"⚠️ Lỗi khi gọi OpenAI API: {e}")
        return None


def _generate_gemini_feedback(mean_diff, motion_var, avg_score):
    prompt = f"""
    Bạn là huấn luyện viên múa Việt Nam.
    Dưới đây là dữ liệu:
    - Độ lệch tư thế: {mean_diff:.3f}
    - Độ mượt chuyển động: {motion_var:.3f}
    - Điểm trung bình: {avg_score:.1f}/100

    Hãy viết nhận xét ngắn gọn, dễ hiểu, thân thiện bằng tiếng Việt:
    - Nhận xét tổng thể
    - Gợi ý cải thiện
    - Câu động viên
    """
    try:
        response = genai_model.generate_content(prompt)
        return [response.text]
    except Exception as e:
        print(f"⚠️ Lỗi khi gọi Gemini API: {e}")
        return None


# ===========================================================
# 4️⃣ Rule-based feedback (fallback)
# ===========================================================
def _generate_rule_based_feedback(mean_diff, motion_var, avg_score):
    feedbacks = []

    # Nhận xét tổng thể
    if avg_score > 90:
        feedbacks.append("🌟 Rất xuất sắc! Các động tác của bạn gần như hoàn hảo.")
    elif avg_score > 75:
        feedbacks.append("👍 Bài múa rất tốt, chỉ cần mượt hơn ở phần tay hoặc hông.")
    elif avg_score > 50:
        feedbacks.append("⚡ Cần cải thiện thêm về độ đều và cảm xúc. Hãy xem lại phần mở đầu.")
    else:
        feedbacks.append("😅 Cần điều chỉnh lại nhịp và tư thế, hãy tập chậm hơn để kiểm soát động tác.")

    # Gợi ý chuyển động
    if motion_var < 0.03:
        feedbacks.append("Động tác hơi cứng, bạn nên di chuyển mềm mại hơn.")
    else:
        feedbacks.append("Chuyển động tự nhiên và có cảm xúc, rất tốt!")

    # Động viên
    feedbacks.append("💪 Tiếp tục luyện tập nhé! Mỗi lần bạn lại tiến bộ thêm.")
    return feedbacks


# ===========================================================
# 5️⃣ Hàm chính: sinh feedback ổn định
# ===========================================================
def generate_feedback(standard_features, user_features, avg_score):
    """
    Sinh phản hồi dựa trên dữ liệu pose.
    - Tự động resample khi độ dài khác nhau.
    - Fallback sang rule-based nếu không có API.
    """
    try:
        len_std, len_user = len(standard_features), len(user_features)
        if len_std == 0 or len_user == 0:
            return ["⚠️ Không đủ dữ liệu để tạo phản hồi. Hãy thử lại với video khác."]

        # ⚖️ Resample để có cùng độ dài
        target_len = min(len_std, len_user)
        if len_std != len_user:
            print(f"⏩ Chuẩn hóa độ dài: {len_std} → {target_len}, {len_user} → {target_len}")
            standard_features = _resample_sequence(standard_features, target_len)
            user_features = _resample_sequence(user_features, target_len)

        # 🧮 Tính độ lệch và độ mượt
        mean_diff = float(np.mean(np.abs(standard_features - user_features)))
        motion_var = float(np.var(user_features))

        # Ưu tiên AI nếu có
        if openai_client:
            print("🤖 Dùng OpenAI GPT để sinh feedback...")
            fb = _generate_openai_feedback(mean_diff, motion_var, avg_score)
            if fb:
                return fb

        if genai_model:
            print("✨ Dùng Gemini để sinh feedback...")
            fb = _generate_gemini_feedback(mean_diff, motion_var, avg_score)
            if fb:
                return fb

        # Fallback
        print("🧠 Dùng mô phỏng AI nội bộ (rule-based).")
        return _generate_rule_based_feedback(mean_diff, motion_var, avg_score)

    except Exception as e:
        print(f"⚠️ Lỗi khi tạo feedback: {e}")
        return ["⚠️ Không thể tạo phản hồi do lỗi xử lý dữ liệu."]
