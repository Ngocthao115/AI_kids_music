import os
import time
import json
import requests
import datetime as dt
import csv
import re
import io
import base64
import unicodedata
from typing import List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ================== 1) CONFIG & ENV ==================
load_dotenv()


def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SUNO_API_KEY = get_secret("SUNO_API_KEY")
SUNO_API_BASE = get_secret("SUNO_API_BASE", "https://api.sunoapi.org")
SUNO_MODEL = get_secret("SUNO_MODEL", "V5")
SUNO_CALLBACK_URL = get_secret("SUNO_CALLBACK_URL")
DEFAULT_SUNOSTYLE = get_secret(
    "DEFAULT_SUNOSTYLE",
    "Children's Pop, cheerful, simple melody, clapping beat, preschool music",
)

# Supabase: để trống vẫn chạy local bình thường
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
SUPABASE_BUCKET = get_secret("SUPABASE_BUCKET", "mp3")

ANALYTICS_CSV = get_secret("ANALYTICS_CSV", "du_lieu_tao_nhac.csv")

if not OPENAI_API_KEY:
    st.error("Thiếu OPENAI_API_KEY — thêm trong Secrets.")
    st.stop()
if not SUNO_API_KEY:
    st.error("Thiếu SUNO_API_KEY — thêm trong Secrets.")
    st.stop()
if not SUNO_CALLBACK_URL:
    st.warning("Chưa có SUNO_CALLBACK_URL — app vẫn có thể hoạt động bằng cách poll kết quả.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
HEADERS = {"Authorization": f"Bearer {SUNO_API_KEY}", "Content-Type": "application/json"}

# Kết nối Supabase (tuỳ chọn)
supabase = None
supabase_status = "❌"
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase_status = "✅"
    except Exception as e:
        st.warning(f"Không khởi tạo được Supabase client: {e}")

# Local output
OUTPUT_DIR = "outputs"
MP3_DIR = os.path.join(OUTPUT_DIR, "mp3")
COVER_DIR = os.path.join(OUTPUT_DIR, "covers")
os.makedirs(MP3_DIR, exist_ok=True)
os.makedirs(COVER_DIR, exist_ok=True)

HISTORY_CSV = os.path.join(OUTPUT_DIR, "tracks.csv")
EXPECTED_HEADER = [
    "time",
    "title",
    "topic",
    "keywords",
    "style",
    "language",
    "verses",
    "bridge",
    "instrumental",
    "track_index",
    "audio_url",
    "image_url",
    "mp3_path",
    "cover_path",
    "lyrics",
    "age_group",
    "theme_month",
    "source_type",  # "new" hoac "poem"
]

# ================== 2) PROMPT HỆ THỐNG ==================
DEFAULT_LYRICS_SYSTEM = (
    "Bạn là một chuyên gia sáng tác nhạc thiếu nhi và là một nhà sư phạm mầm non giàu kinh nghiệm, "
    "bạn hiểu rõ tâm lý trẻ em từ 3-6 tuổi và có khả năng biến các bài học giáo dục thành lời ca trong sáng, dễ thuộc, dễ nhớ. "
    "Hãy sáng tác lời cho một bài hát thiếu nhi. Lời bài hát phải tuân thủ nghiêm ngặt các tiêu chuẩn kỹ thuật và nội dung như mỗi câu hát chỉ được phép có từ 5 đến 10 từ. "
    "Ngôn ngữ và nội dung phải mang tính giáo dục, ý nghĩa nhân văn, dạy trẻ về thế giới xung quanh, thói quen tốt hoặc tình yêu thương. "
    "Sử dụng từ ngữ vui tươi hoặc tình cảm, rộn ràng, giàu hình ảnh, vần điệu rõ ràng, có điệp khúc dễ nhớ. "
    "YÊU CẦU ĐỊNH DẠNG: Mặc định viết dạng bài hát thiếu nhi thông thường với các nhãn [Verse] và [Chorus]. "
    "CHỈ KHI trong yêu cầu của người dùng có nêu rõ phong cách 'rap' hoặc 'rap thiếu nhi' thì mới viết theo cấu trúc rap: "
    "[Intro], [Spoken], [Melodic Chorus], [Verse 1], [Hook/Chorus], [Verse 2], [Outro], nhịp câu ngắn, gieo vần dễ đọc, không dùng từ người lớn. "
    "Nếu không có yêu cầu rap thì tuyệt đối không dùng cấu trúc rap."
)

# ================== 3) HÀM NGHIỆP VỤ ==================

def build_user_prompt(
    topic: str,
    language: str = "vi",
    target_words: Optional[List[str]] = None,
    verses: int = 2,
    include_bridge: bool = True,
    min_lines: int = 12,
    max_lines: int = 18,
) -> str:
    tw = ", ".join(target_words) if target_words else "Không bắt buộc"
    structure = ["- Cấu trúc: [Verse 1] → [Chorus]"]
    for i in range(2, verses + 1):
        structure.append(f"→ [Verse {i}] → [Chorus]")
    if include_bridge:
        structure.append("→ [Bridge] (ngắn 2–4 dòng) → [Chorus] (kết)")
    return (
        f"Chủ đề: {topic}\n"
        f"Ngôn ngữ: {language}\n"
        "Yêu cầu:\n"
        "- Ngôn ngữ đơn giản, an toàn cho trẻ 3–6 tuổi; tích cực, hồn nhiên.\n"
        "- Vần điệu rõ, nhịp vui tươi hoặc tình cảm nhẹ nhàng, câu ngắn.\n"
        f"{' '.join(structure)}.\n"
        f"- Từ ngữ chính (nếu lồng được): {tw}\n"
        f"- Độ dài ~{min_lines}–{max_lines} dòng.\n"
        "- Định dạng đầu ra có nhãn [Verse]/[Chorus]/[Bridge].\n"
    )



def generate_lyrics(
    topic: str,
    target_words: Optional[List[str]] = None,
    language: str = "vi",
    verses: int = 2,
    bridge: bool = True,
) -> str:
    user_prompt = build_user_prompt(
        topic,
        language=language,
        target_words=target_words,
        verses=verses,
        include_bridge=bridge,
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DEFAULT_LYRICS_SYSTEM},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.9,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()



def refine_lyrics(original_text: str, instruction: str = "") -> str:
    if not original_text.strip():
        return original_text
    user_msg = (
        "Hãy chỉnh sửa lời bài hát thiếu nhi bên dưới, giữ nguyên chủ đề và tinh thần cho trẻ từ 3 đến 6 tuổi. "
        "Tăng vần điệu, nhịp mượt, chia đoạn rõ [Verse]/[Chorus]/[Bridge]. "
        "Áp dụng nhẹ nhàng chỉ dẫn nếu có, không kéo quá dài.\n\n"
        f"Chỉ dẫn: {instruction or 'Không có'}\n\n"
        "Văn bản cần chỉnh:\n" + original_text
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DEFAULT_LYRICS_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def poem_to_song(poem_text: str, age_group: str = "Mẫu giáo (3-6 tuổi)", style_hint: str = "") -> str:
    """Chuyển bài thơ / câu chuyện thành lời bài hát thiếu nhi."""
    system = (
        "Bạn là chuyên gia chuyển thơ và truyện thiếu nhi thành lời bài hát. "
        "Giữ nguyên tinh thần, hình ảnh và thông điệp của bài thơ/câu chuyện gốc. "
        "Chuyển thành lời bài hát có cấu trúc [Verse]/[Chorus]/[Bridge], vần điệu rõ, nhịp nhàng. "
        "Ngôn ngữ phù hợp lứa tuổi mầm non, câu ngắn 5-10 từ, dễ hát dễ nhớ."
    )
    user = (
        f"Lứa tuổi: {age_group}\n"
        f"Phong cách nhạc gợi ý: {style_hint or 'Tươi vui, trẻ em'}\n\n"
        f"Nội dung gốc:\n{poem_text}\n\n"
        "Hãy chuyển thành lời bài hát thiếu nhi với cấu trúc [Verse 1] → [Chorus] → [Verse 2] → [Chorus] → [Bridge] → [Chorus]. "
        "Giữ nguyên ý nghĩa, hình ảnh đẹp của bài gốc. Viết bằng tiếng Việt."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def get_topic_suggestions(chu_de: str) -> str:
    """Gợi ý từ khóa và nội dung cho chủ đề được chọn."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia giáo dục mầm non Việt Nam."},
            {"role": "user", "content": (
                f"Cho chủ đề '{chu_de}' trong chương trình giáo dục mầm non, "
                "hãy gợi ý ngắn gọn: 1) Mô tả bài hát phù hợp (1 câu), "
                "2) 5-7 từ khóa chính, 3) Mục tiêu giáo dục (1 câu). "
                "Trả lời ngắn, thực tế."
            )},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def generate_stats_summary(df) -> dict:
    """Tạo thống kê tóm tắt từ dataframe lịch sử."""
    if df is None or len(df) == 0:
        return {}
    stats = {
        "tong_bai": len(df),
        "chu_de_pho_bien": "",
        "phong_cach_pho_bien": "",
        "thang_nhieu_nhat": "",
    }
    if "topic" in df.columns:
        top_topic = df["topic"].value_counts().head(1)
        if len(top_topic) > 0:
            stats["chu_de_pho_bien"] = top_topic.index[0]
    if "style" in df.columns:
        top_style = df["style"].value_counts().head(1)
        if len(top_style) > 0:
            stats["phong_cach_pho_bien"] = top_style.index[0]
    return stats


def _friendly_suno_error(response: requests.Response) -> RuntimeError:
    try:
        data = response.json()
        msg = data.get("msg") or data.get("message") or response.text
    except Exception:
        msg = response.text

    if response.status_code == 429:
        return RuntimeError(
            "Suno/API đang từ chối yêu cầu vì hết credit hoặc vượt quota. "
            f"Chi tiết: {msg}"
        )
    if response.status_code == 401:
        return RuntimeError("SUNO_API_KEY không hợp lệ hoặc đã hết hiệu lực.")
    if response.status_code == 403:
        return RuntimeError("Yêu cầu bị từ chối. Kiểm tra quyền truy cập API Suno.")
    return RuntimeError(f"Suno request failed ({response.status_code}): {msg}")



def suno_generate_song(prompt: str, title: str, style: str, instrumental: bool = False) -> str:
    endpoint = f"{SUNO_API_BASE}/api/v1/generate"
    payload = {
        "prompt": prompt[:1800],
        "title": title[:64],
        "style": style[:200],
        "model": SUNO_MODEL,
        "instrumental": instrumental,
        "customMode": True,
        "callBackUrl": SUNO_CALLBACK_URL,
    }
    r = requests.post(endpoint, headers=HEADERS, json=payload, timeout=60)
    if r.status_code >= 400:
        raise _friendly_suno_error(r)
    data = r.json()
    if data.get("code") != 200 or not data.get("data", {}).get("taskId"):
        raise RuntimeError("Suno generate failed: " + json.dumps(data, ensure_ascii=False))
    return data["data"]["taskId"]



def suno_poll(task_id: str, timeout_sec: int = 360, interval_sec: int = 8):
    endpoint = f"{SUNO_API_BASE}/api/v1/generate/record-info"
    started = time.time()
    last_msg = ""
    while time.time() - started < timeout_sec:
        r = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {SUNO_API_KEY}"},
            params={"taskId": task_id},
            timeout=60,
        )
        if r.status_code >= 400:
            raise _friendly_suno_error(r)

        data = r.json()
        last_msg = data.get("msg", "") if isinstance(data, dict) else ""
        try:
            items = data["data"]["response"]["sunoData"]
            ready = [it for it in items if it.get("audioUrl") or it.get("audioUrlHigh")]
            if ready:
                return ready
        except Exception:
            pass
        time.sleep(interval_sec)
    raise TimeoutError(f"Hết thời gian chờ trả kết quả. {last_msg}".strip())



def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content



def ascii_slugify(text: str) -> str:
    text = (text or "").strip().replace(" ", "_")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    text = text.strip("._-") or "file"
    return text[:80]



def ensure_history_schema():
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(EXPECTED_HEADER)
        return

    with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header == EXPECTED_HEADER:
        return

    rows_old = []
    with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_old.append(row)

    tmp = HISTORY_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED_HEADER)
        w.writeheader()
        for old in rows_old:
            newrow = {k: old.get(k, "") for k in EXPECTED_HEADER}
            w.writerow(newrow)
    os.replace(tmp, HISTORY_CSV)



def sb_upload_bytes(bucket: str, path: str, data_bytes: bytes, content_type: str) -> Optional[str]:
    """Upload bytes len Supabase Storage va tra ve public URL."""
    if not supabase or not data_bytes:
        return None
    try:
        supabase.storage.from_(bucket).upload(
            path=path,
            file=data_bytes,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    except Exception as upload_err:
        err_msg = str(upload_err)
        if "already exists" in err_msg or "Duplicate" in err_msg or "23505" in err_msg:
            try:
                supabase.storage.from_(bucket).remove([path])
                supabase.storage.from_(bucket).upload(
                    path=path,
                    file=data_bytes,
                    file_options={"content-type": content_type},
                )
            except Exception as retry_err:
                st.warning(f"Upload Supabase that bai ({path}): {retry_err}")
                return None
        else:
            st.warning(f"Upload Supabase that bai ({path}): {upload_err}")
            return None
    try:
        pub = supabase.storage.from_(bucket).get_public_url(path)
        if isinstance(pub, str):
            return pub
        if isinstance(pub, dict):
            return pub.get("publicUrl") or pub.get("data", {}).get("publicUrl") or ""
        return str(pub)
    except Exception as url_err:
        st.warning(f"Khong lay duoc public URL ({path}): {url_err}")
        return None



def sb_upload_cover(bucket: str, path: str, img_bytes: bytes) -> Optional[str]:
    """Upload ảnh bìa lên Supabase Storage. Tự detect content-type từ đuôi file."""
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    return sb_upload_bytes(bucket, path, img_bytes, mime)


def sb_save_track_metadata(row: dict) -> bool:
    """Lưu metadata bài hát vào bảng 'tracks' trong Supabase Database (tuỳ chọn)."""
    if not supabase:
        return False
    try:
        supabase.table("tracks").upsert(row, on_conflict="time,track_index").execute()
        return True
    except Exception as e:
        # Bảng chưa tồn tại hoặc lỗi khác — bỏ qua, không crash app
        return False


def load_history_from_supabase() -> Optional[pd.DataFrame]:
    """
    Tải toàn bộ lịch sử bài hát từ Supabase Database.
    Trả về DataFrame hoặc None nếu không khả dụng.
    """
    if not supabase:
        return None
    try:
        res = supabase.table("tracks").select("*").order("time", desc=True).execute()
        rows = res.data if hasattr(res, "data") else []
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # Đảm bảo các cột chuẩn tồn tại
        for col in EXPECTED_HEADER:
            if col not in df.columns:
                df[col] = ""
        return df[EXPECTED_HEADER]
    except Exception:
        return None



def write_history_row(row: dict) -> None:
    ensure_history_schema()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED_HEADER)
        w.writerow({k: row.get(k, "") for k in EXPECTED_HEADER})



def load_history_df_local() -> pd.DataFrame:
    ensure_history_schema()
    try:
        return pd.read_csv(HISTORY_CSV, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame(columns=EXPECTED_HEADER)



def log_prompt_to_csv(row: dict, csv_path: str = ANALYTICS_CSV):
    mapped = {
        "Thời gian": row.get("time", ""),
        "Tên bài hát": row.get("title", ""),
        "Miêu tả bài hát": row.get("topic", ""),
        "Từ ngữ gợi ý": row.get("keywords", ""),
        "Chủ đề / Gợi ý": row.get("topic", ""),
        "Phong cách nhạc": row.get("style", ""),
        "Ngôn ngữ": row.get("language", ""),
        "Số verse": row.get("verses", ""),
        "Bridge": row.get("bridge", ""),
        "Liên kết nhạc": row.get("audio_url", ""),
        "Liên kết lời hát": row.get("lyrics_url", ""),
    }
    cols = [
        "Thời gian",
        "Tên bài hát",
        "Miêu tả bài hát",
        "Từ ngữ gợi ý",
        "Chủ đề / Gợi ý",
        "Phong cách nhạc",
        "Ngôn ngữ",
        "Số verse",
        "Bridge",
        "Liên kết nhạc",
        "Liên kết lời hát",
    ]

    if os.path.exists(csv_path):
        base = pd.read_csv(csv_path)
    else:
        base = pd.DataFrame(columns=cols)

    for c in cols:
        if c not in base.columns:
            base[c] = None

    key1, key2 = "Tên bài hát", "Liên kết nhạc"
    base[key1] = base[key1].fillna("").astype(str).str.strip()
    base[key2] = base[key2].fillna("").astype(str).str.strip()
    new_key = f"{(mapped[key1] or '').strip()}||{(mapped[key2] or '').strip()}"

    base["_key"] = base[key1] + "||" + base[key2]
    if new_key in set(base["_key"].tolist()):
        idx = base.index[base["_key"] == new_key][0]
        for c in cols:
            val = mapped.get(c, "")
            if str(val).strip() != "":
                base.at[idx, c] = val
    else:
        row_df = pd.DataFrame([[mapped.get(c, "") for c in cols]], columns=cols)
        row_df["_key"] = new_key
        base = pd.concat([base, row_df], ignore_index=True)

    out = base.drop(columns=["_key"], errors="ignore")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")



def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", str(s)) if unicodedata.category(ch) != "Mn")



def norm_txt(s: str) -> str:
    return strip_accents(str(s).lower().strip())



def parse_time_safe(x):
    s = "" if x is None else str(x).strip()
    fmts = [
        "%Y%m%d-%H%M%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ]
    for fmt in fmts:
        try:
            return dt.datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT



def show_cover_from_row(row: pd.Series):
    cover_path = str(row.get("cover_path", "") or "").strip()
    image_url = str(row.get("image_url", "") or "").strip()
    if cover_path and os.path.exists(cover_path):
        st.image(cover_path, use_container_width=True)
    elif image_url:
        st.image(image_url, use_container_width=True)
    else:
        st.image("https://picsum.photos/seed/kidsmusic/600/400", use_container_width=True)



def show_audio_from_row(row: pd.Series, key_suffix: str = ""):
    mp3_path = str(row.get("mp3_path", "") or "").strip()
    audio_url = str(row.get("audio_url", "") or "").strip()
    if mp3_path and os.path.exists(mp3_path):
        with open(mp3_path, "rb") as f:
            data = f.read()
        st.audio(data, format="audio/mp3")
        st.download_button(
            "⬇ Tải MP3",
            data=data,
            file_name=os.path.basename(mp3_path),
            mime="audio/mpeg",
            use_container_width=True,
            key=f"dl_{row.get('time','')}_{row.get('track_index','1')}_{key_suffix}",
        )
    elif audio_url:
        st.audio(audio_url, format="audio/mp3")

# ================== 4) UI / THEME ==================
st.set_page_config(page_title="NHẠC AI THIẾU NHI - MẦM NON", page_icon="🎵", layout="centered")


def add_bg_from_local(image_path: str, alpha: float = 0.85, size: str = "1300px auto", position: str = "top center"):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image:
                    linear-gradient(rgba(255,255,255,{alpha}), rgba(255,255,255,{alpha})),
                    url("data:image/png;base64,{encoded}");
                background-size: {size};
                background-position: {position};
                background-repeat: no-repeat;
            }}
            .stTextInput, .stSelectbox {{
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Không tải được ảnh nền: {e}")


add_bg_from_local("music2.jpg", alpha=0.88, size="cover", position="center")

st.markdown(
    """
<style>
:root{
  --bg: #FFFFFF;
  --bg-soft: #FFFFFF;
  --primary: #FFB996;
  --primary-strong: #FFA97A;
  --chip: #FFE8D9;
  --input: #F8FAFD;
  --text: #2D2D2D;
  --muted: #6B7280;
  --ring: #FFD7C3;
  --shadow: 0 10px 20px rgba(17,24,39,.07);
}
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&display=swap');
html, body, .stApp, [class*="css"]{
  font-family: 'Fredoka', system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;
  color: var(--text) !important;
}
.stApp{ background: var(--bg-soft) !important; }
.main .block-container{
  background: var(--bg) !important;
  border-radius: 16px !important;
  padding: 2rem !important;
  box-shadow: var(--shadow);
}
section[data-testid="stSidebar"] > div{ background: var(--chip) !important; }
section[data-testid="stSidebar"] *{ color: var(--text) !important; }
h1,h2,h3{ color: var(--text) !important; }
hr{ border: none; height:1px; background: var(--ring); }
.stTextInput textarea, .stTextArea textarea,
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input, .stDateInput input{
  background: var(--input) !important;
  border: 1px solid var(--ring) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
}
.stTextInput:focus-within input,
.stTextArea:focus-within textarea,
.stSelectbox:focus-within div[data-baseweb="select"] > div{
  box-shadow: 0 0 0 3px var(--ring) !important;
}
::placeholder{ color: rgba(45,45,45,.45) !important; }
button[kind="primary"],
button[data-testid="baseButton-primary"],
div.stButton > button,
div.stDownloadButton > button{
  background: var(--primary) !important;
  border-color: var(--primary) !important;
  color: #fff !important;
  border-radius: 12px !important;
  padding: .6rem 1.2rem !important;
  font-weight: 600 !important;
  font-size: 16px !important;
  border: none !important;
  box-shadow: 0 6px 12px rgba(255,185,150,.35) !important;
  transition: transform .15s ease, filter .15s ease;
}
button:hover{ background: var(--primary-strong) !important; transform: translateY(-1px); }
button[data-testid="baseButton-secondary"]{ background: var(--chip) !important; color: var(--text) !important; }
div[data-baseweb="tab-highlight"]{ background: var(--primary) !important; }
div[data-baseweb="tab-border"]{ background: transparent !important; }
.badge{
  display:inline-flex; align-items:center; gap:.35rem;
  padding:.35rem .7rem; border-radius:999px;
  background: var(--chip); color: var(--text); font-weight:600;
}
[data-baseweb="slider"] div[role="slider"]{ background: var(--primary) !important; }
[data-baseweb="slider"] div[aria-hidden="true"]{ background: var(--ring) !important; }
.stProgress > div > div{ background: var(--primary) !important; }
.stTabs div[data-baseweb="tab-highlight"],
.stTabs div[data-baseweb="tab-border"]{ display: none !important; }
.stTabs div[data-baseweb="tab-list"]{ gap: .5rem !important; border-bottom: none !important; }
.stTabs button[role="tab"],
.stTabs div[data-baseweb="tab"] > button{
  background: var(--chip) !important;
  color: var(--text) !important;
  border: 1px solid var(--ring) !important;
  border-radius: 999px !important;
  padding: .4rem .9rem !important;
  font-weight: 600 !important;
}
.stTabs button[role="tab"][aria-selected="true"]{
  background: var(--primary) !important;
  color: #fff !important;
  border-color: var(--primary) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.stTextInput label, .stSelectbox label, .stNumberInput label,
div[data-testid="stTextInputLabel"], div[data-testid="stSelectboxLabel"],
div[data-testid="stNumberInputLabel"], div[data-testid="stMarkdownContainer"] p {
    font-weight: 700 !important;
    color: #2c1e1e !important;
    font-size: 16px !important;
    letter-spacing: 0.3px;
}
.stTextInput div[data-baseweb="input"],
.stSelectbox div[data-baseweb="select"],
.stNumberInput div[data-baseweb="input"] {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    border: 1px solid #f7d7c3;
}
.stTextInput, .stSelectbox, .stNumberInput { margin-bottom: 0.8rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# State
st.session_state.setdefault("lyrics", "")
st.session_state.setdefault("title", "")
st.session_state.setdefault("topic", "")
st.session_state.setdefault("targets", [])
st.session_state.setdefault("generated", False)
st.session_state.setdefault("keywords", "")
st.session_state.setdefault("language", "vi")
st.session_state.setdefault("verses", 2)
st.session_state.setdefault("bridge", True)

# Sidebar

# Chủ đề theo tháng - Chương trình GDMN Bộ GD&ĐT
CHU_DE_THANG = {
    "Tháng 9 — Trường mầm non": {"mo_ta": "Trường mầm non thân yêu của bé", "tu_khoa": "Trường lớp, cô giáo, bạn bè, sân chơi, đồ chơi"},
    "Tháng 10 — Bản thân": {"mo_ta": "Cơ thể bé và những điều bé thích", "tu_khoa": "Tay chân, mặt mũi, tên bé, sở thích, cảm xúc"},
    "Tháng 11 — Gia đình": {"mo_ta": "Gia đình yêu thương của bé", "tu_khoa": "Ba mẹ, ông bà, anh chị em, ngôi nhà, yêu thương"},
    "Tháng 12 — Nghề nghiệp": {"mo_ta": "Các nghề nghiệp trong xã hội", "tu_khoa": "Bác sĩ, cô giáo, chú công an, nông dân, kỹ sư"},
    "Tháng 1-2 — Thế giới động vật": {"mo_ta": "Các con vật gần gũi với bé", "tu_khoa": "Chó mèo, gà vịt, bướm sâu, rừng núi, biển cả"},
    "Tháng 3 — Thế giới thực vật": {"mo_ta": "Cây cối hoa lá xung quanh bé", "tu_khoa": "Hoa lá, cây xanh, rau củ, mùa xuân, vườn cây"},
    "Tháng 3-4 — Phương tiện giao thông": {"mo_ta": "Các phương tiện giao thông bé biết", "tu_khoa": "Xe ô tô, xe đạp, máy bay, tàu thuyền, an toàn"},
    "Tháng 4-5 — Quê hương Đất nước": {"mo_ta": "Tình yêu quê hương đất nước Việt Nam", "tu_khoa": "Quê hương, Việt Nam, cờ đỏ, sao vàng, biển đảo"},
    "Tháng 5-6 — Mùa hè": {"mo_ta": "Mùa hè vui tươi và kỳ nghỉ hè", "tu_khoa": "Nắng vàng, biển xanh, kem mát, bướm hoa, nghỉ hè"},
}

AGE_GROUPS = [
    "Nhà trẻ (0-3 tuổi)",
    "Mẫu giáo bé (3-4 tuổi)",
    "Mẫu giáo nhỡ (4-5 tuổi)",
    "Mẫu giáo lớn (5-6 tuổi)",
]

LINH_VUC = [
    "🏃 Phát triển thể chất",
    "🧠 Phát triển nhận thức",
    "🗣️ Phát triển ngôn ngữ",
    "🎨 Phát triển thẩm mỹ",
    "❤️ Phát triển tình cảm - Xã hội",
]

with st.sidebar:
    st.markdown("## 👩‍🏫 Hướng dẫn nhanh")
    st.markdown(
        "**✨ Tạo bài hát mới:**\n"
        "1. Chọn chủ đề tháng → bấm **Tự động điền**\n"
        "2. Chọn độ tuổi, phong cách nhạc\n"
        "3. Bấm **Tạo lời** → chỉnh sửa nếu cần\n"
        "4. Bấm **Tạo nhạc** → tải MP3\n\n"
        "**📖 Từ thơ/truyện:**\n"
        "1. Vào tab **Thơ/Truyện → Nhạc**\n"
        "2. Dán nội dung vào ô\n"
        "3. Bấm **Chuyển thành bài hát**\n"
        "4. Bấm **Tạo nhạc** → tải MP3"
    )
    st.divider()
    st.markdown("**📅 Chủ đề theo tháng:**")
    for thang in list(CHU_DE_THANG.keys())[:4]:
        st.caption(f"• {thang}")
    st.caption("• ...")
    st.divider()
    st.caption(f"Model Suno: **{SUNO_MODEL}**")
    st.caption(f"Supabase: **{supabase_status}**")

# Header
st.title("🎵 Kids Song AI")
st.markdown(
    '<span class="badge">🏫 Dành riêng cho Giáo viên Mầm non</span>&nbsp;'
    '<span class="badge">✨ Tạo nhạc thiếu nhi bằng AI</span>&nbsp;'
    '<span class="badge">📖 Chuyển Thơ/Truyện → Bài Hát</span>',
    unsafe_allow_html=True
)

# Tabs
tab_make, tab_poem, tab_library, tab_stats, tab_history, tab_settings = st.tabs(
    ["✨ Tạo bài hát", "📖 Thơ/Truyện → Nhạc", "📚 Thư viện", "📊 Thống kê", "🗂️ Lịch sử", "⚙️ Cài đặt"]
)


# ================== TAB 1: TẠO BÀI HÁT ==================
with tab_make:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Chọn chủ đề theo tháng
    st.markdown("#### 📅 Chọn chủ đề theo chương trình GDMN")
    col_theme, col_auto = st.columns([3, 1])
    with col_theme:
        selected_theme = st.selectbox(
            "Chủ đề tháng (theo Chương trình GDMN Bộ GD&ĐT)",
            ["— Tự nhập —"] + list(CHU_DE_THANG.keys()),
            key="theme_select"
        )
    with col_auto:
        btn_autofill = st.button("✨ Tự động điền", use_container_width=True, key="btn_autofill")

    if btn_autofill and selected_theme != "— Tự nhập —":
        info = CHU_DE_THANG[selected_theme]
        st.session_state.topic = info["mo_ta"]
        st.session_state.keywords = info["tu_khoa"]
        st.session_state.title = info["mo_ta"]
        st.rerun()

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Miêu tả bài hát", st.session_state.topic or "Trường mầm non của bé")
        target_str = st.text_input(
            "Từ ngữ gợi ý (phân tách bởi dấu phẩy)",
            st.session_state.keywords or "Đồ chơi, sân trường, lớp học, thân thương",
        )
        title = st.text_input("Tiêu đề bài hát", st.session_state.title or "Trường mầm non của bé")
    with col2:
        verses = st.number_input("Số verse", 1, 4, int(st.session_state.verses))
        bridge = st.toggle("Thêm Bridge", value=bool(st.session_state.bridge))
        language = st.selectbox(
            "Ngôn ngữ",
            ["Vi", "En"],
            index=0 if st.session_state.language == "vi" else 1,
        )
        age_group = st.selectbox("Độ tuổi", AGE_GROUPS, index=1, key="age_group_tab1")
        linh_vuc = st.selectbox("Lĩnh vực phát triển", LINH_VUC, index=3, key="linh_vuc_tab1")

    st.session_state.topic = topic
    st.session_state.title = title
    st.session_state.keywords = target_str
    st.session_state.language = "vi" if str(language).lower().startswith("v") else "en"
    st.session_state.verses = int(verses)
    st.session_state.bridge = bool(bridge)

    STYLE_DISPLAY = [
        "Children's Pop – Nhạc pop thiếu nhi, tươi vui, dễ hát",
        "Playful / Upbeat Kids – Nhạc vui nhộn, hoạt bát, sinh động",
        "Nursery Rhymes – Đồng dao, hát thiếu nhi cổ điển",
        "Educational Songs – Nhạc học tập, dạy chữ, đếm số",
        "Children's Folk – Dân ca thiếu nhi, nhẹ nhàng, gần gũi",
        "Lullaby – Nhạc ru, dễ ngủ, êm dịu",
        "Magical / Whimsical Kids – Huyền ảo, cổ tích, tưởng tượng",
        "Children's Jazz – Nhạc jazz nhẹ, thư giãn, tinh tế",
        "Musical Story / Narrative – Nhạc kể chuyện, diễn cảm",
        "Children's Rock",
        "Children's Rap",
    ]

    STYLE_MAP = {
        "Children's Pop": "children pop, cheerful, simple melody, clapping beat, preschool music",
        "Playful / Upbeat Kids": "upbeat kids music, playful, dancing rhythm, handclap, xylophone",
        "Nursery Rhymes": "nursery rhyme style, simple melody, repetitive chorus, bright and soft",
        "Educational Songs": "educational kids song, counting and alphabet style, cheerful tempo, clear vocals",
        "Children's Folk": "Vietnamese Folk song, traditional rural melody, emotional and nostalgic, soft vocals, simple rhythm, instruments like đàn bầu, sáo trúc, and gentle percussion",
        "Lullaby": "lullaby for children, soft piano and harp, gentle tempo, dreamy atmosphere, low dynamics",
        "Magical / Whimsical Kids": "whimsical kids music, magical fantasy, bells and strings, dreamy vibe",
        "Children's Jazz": "soft jazz for kids, swing rhythm, piano and brushes, happy mood",
        "Musical Story / Narrative": "storytelling kids song, expressive melody, dynamic flow, cinematic tone",
        "Children's Rock": (
            "Vietnamese children's rock song, very simple and playful. "
            "Style: soft children's rock, not adult rock. Tempo: medium, steady, marching feel. "
            "Guitar: clean electric guitar, very light distortion, simple strumming only. "
            "Melody: strong, easy-to-sing melody, limited pitch range. Vocals: friendly child-like singing voice, not shouting. "
            "Structure: verse – catchy chorus – verse – chorus. Mood: happy, energetic, safe for preschool kids. "
            "NO heavy drums, NO loud distortion, NO aggressive rock style."
        ),
        "Children's Rap": (
            "Vietnamese children's rap song for preschool kids, chant-style rap with clear melody. "
            "Spoken-sung rap, narrow pitch range, simple melodic contour. Tempo: medium, steady, easy for children to clap along. "
            "Beat: very light hip-hop beat, soft kick and clap, no heavy bass, no trap elements. "
            "Vocals: cheerful child-like voice, smiling tone, clear pronunciation, slow rap speed. "
            "Structure: intro chant – verse – melodic chorus – verse – chorus. "
            "Lyrics style: short sentences, repetition, call-and-response. STRICTLY NO adult rap style, no fast flow, no aggressive rhythm, no slang."
        ),
    }

    style_display = st.selectbox("Phong cách nhạc", STYLE_DISPLAY, index=0)
    style_key = style_display.split("–")[0].strip()
    style_for_api = STYLE_MAP.get(style_key, DEFAULT_SUNOSTYLE)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 TIẾN TRÌNH SÁNG TÁC")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        btn_generate = st.button("✨ Tạo lời bài hát", use_container_width=True)
    with c2:
        refine_hint = st.text_input(
            "Chỉ dẫn refine (tuỳ chọn)",
            placeholder="Ví dụ: nhịp nhanh hơn, thêm điệp khúc…",
        )
    with c3:
        btn_refine = st.button(
            "🪄 Refine",
            use_container_width=True,
            disabled=not bool(st.session_state.lyrics.strip()),
        )

    if btn_generate:
        try:
            targets = [w.strip() for w in target_str.split(",") if w.strip()]
            with st.spinner("Đang sáng tác lời..."):
                lyrics = generate_lyrics(
                    topic,
                    targets,
                    language=st.session_state.language,
                    verses=int(verses),
                    bridge=bool(bridge),
                )
            st.session_state.lyrics = lyrics
            st.session_state.title = title
            st.session_state.topic = topic
            st.session_state.targets = targets
            st.session_state.generated = True
            st.success("Đã sinh lời. Chỉnh sửa trực tiếp hoặc bấm Refine.")
        except Exception as e:
            st.error(str(e))

    if btn_refine and st.session_state.lyrics.strip():
        try:
            with st.spinner("Đang chỉnh sửa lời..."):
                st.session_state.lyrics = refine_lyrics(st.session_state.lyrics, refine_hint)
            st.success("Đã refine lời bài hát.")
        except Exception as e:
            st.error(str(e))

    st.session_state.lyrics = st.text_area(
        "Soạn thảo/Chỉnh sửa tại đây trước khi tạo nhạc:",
        value=st.session_state.lyrics,
        height=320,
    )
    new_title = st.text_input("🔤Tên bài hát sau khi refine", st.session_state.title)
    if new_title.strip():
        st.session_state.title = new_title.strip()

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        instrumental = st.toggle("Chỉ giai điệu (instrumental)", value=False)
    with right:
        btn_music = st.button(
            "🎧 Tạo nhạc",
            use_container_width=True,
            disabled=not bool(st.session_state.lyrics.strip()),
        )

    if btn_music and st.session_state.lyrics.strip():
        try:
            with st.spinner("Đang tạo bài hát..."):
                task_id = suno_generate_song(
                    st.session_state.lyrics,
                    st.session_state.title or "Kids Song",
                    style=style_for_api,
                    instrumental=instrumental,
                )
                tracks = suno_poll(task_id)

            st.subheader("🎧 Kết quả")
            ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            base = ascii_slugify(st.session_state.title or "Kids_Song")

            for i, t in enumerate(tracks, 1):
                audio_url_orig = t.get("audioUrlHigh") or t.get("audioUrl") or ""
                image_url_orig = t.get("imageUrl") or ""
                mp3_path = ""
                cover_path = ""
                audio_bytes = b""

                if audio_url_orig:
                    audio_bytes = download_bytes(audio_url_orig)
                    mp3_path = os.path.join(MP3_DIR, f"{ts}_{i}_{base}.mp3")
                    with open(mp3_path, "wb") as f:
                        f.write(audio_bytes)

                if image_url_orig:
                    try:
                        img_bytes = download_bytes(image_url_orig)
                        cover_path = os.path.join(COVER_DIR, f"{ts}_{i}_{base}.jpg")
                        with open(cover_path, "wb") as f:
                            f.write(img_bytes)
                    except Exception as e:
                        st.warning(f"Không lưu được ảnh bìa cho bản {i}: {e}")
                        cover_path = ""

                # ── Upload MP3 lên Supabase Storage ──
                audio_url_pub = None
                if audio_bytes:
                    audio_url_pub = sb_upload_bytes(
                        SUPABASE_BUCKET,
                        f"mp3/{ts}_{i}_{base}.mp3",
                        audio_bytes,
                        "audio/mpeg",
                    )


                # Anh bia: chi luu local, KHONG upload len Supabase
                audio_url_final = audio_url_pub or audio_url_orig or ""
                image_url_final = image_url_orig  # Dung URL goc tu Suno

                k1, k2 = st.columns([1, 2])
                with k1:
                    if cover_path and os.path.exists(cover_path):
                        st.image(cover_path, use_container_width=True)
                    elif image_url_final:
                        st.image(image_url_final, use_container_width=True)
                with k2:
                    st.write(f"**{st.session_state.title or 'Kids Song'} — Bản {i}**")
                    if audio_url_pub:
                        st.caption(f"☁️ Đã lưu lên Supabase")
                    if mp3_path and os.path.exists(mp3_path):
                        with open(mp3_path, "rb") as f:
                            mp3_data = f.read()
                        st.audio(mp3_data, format="audio/mp3")
                        st.download_button(
                            "⬇️ Tải MP3",
                            data=mp3_data,
                            file_name=os.path.basename(mp3_path),
                            mime="audio/mpeg",
                            use_container_width=True,
                            key=f"dl_now_{ts}_{i}",
                        )
                    elif audio_url_final:
                        st.audio(audio_url_final, format="audio/mp3")

                row = {
                    "time": ts,
                    "title": st.session_state.title or "Kids Song",
                    "topic": st.session_state.topic,
                    "keywords": st.session_state.keywords,
                    "style": style_display,
                    "language": st.session_state.language,
                    "verses": st.session_state.verses,
                    "bridge": st.session_state.bridge,
                    "instrumental": instrumental,
                    "track_index": i,
                    "audio_url": audio_url_final,
                    "image_url": "",
                    "mp3_path": mp3_path if not audio_url_pub else "",
                    "cover_path": "",
                    "lyrics": st.session_state.lyrics,
                    "age_group": st.session_state.get("age_group_tab1", "Mẫu giáo bé (3-4 tuổi)"),
                    "theme_month": st.session_state.get("theme_select", ""),
                    "source_type": "new",
                }
                write_history_row(row)
                log_prompt_to_csv(row)
                # Lưu metadata vào Supabase Database (nếu có bảng tracks)
                sb_save_track_metadata(row)

            st.balloons()
            sb_msg = ""
            if supabase:
                sb_msg = " và đồng bộ lên **Supabase Storage** ☁️"
            st.success(f"Đã lưu bài hát vào local{sb_msg}. Xem ở tab 📚 Thư viện.")

        except Exception as e:
            st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)

# ================== TAB 2: THƠ/TRUYỆN → NHẠC ==================
with tab_poem:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📖 Chuyển Thơ / Câu Chuyện → Bài Hát Thiếu Nhi")
    st.info(
        "💡 **Tính năng độc đáo:** Dán bài thơ, đoạn truyện hoặc nội dung bài học vào đây — "
        "AI sẽ tự động chuyển thành lời bài hát phù hợp với trẻ mầm non, "
        "giữ nguyên tinh thần và thông điệp của bài gốc."
    )

    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        poem_input = st.text_area(
            "📝 Dán nội dung thơ / truyện / bài học vào đây:",
            height=220,
            placeholder="Ví dụ:\nChú thỏ trắng lông như bông\nChú thỏ trắng tai dài vểnh lên\nMắt hồng hồng miệng chúm chím cười\nÂu yếm nhìn ta mỗi buổi sáng tươi...",
            key="poem_input"
        )
        poem_title = st.text_input("Tiêu đề bài hát (sau khi chuyển đổi)", key="poem_title", placeholder="Chú Thỏ Trắng")
    with col_p2:
        poem_age = st.selectbox("Độ tuổi", AGE_GROUPS, index=1, key="poem_age")
        poem_style = st.selectbox("Phong cách nhạc", STYLE_DISPLAY, index=0, key="poem_style")
        st.markdown("---")
        st.markdown("**Ví dụ nội dung phù hợp:**")
        st.caption("✅ Bài thơ thiếu nhi")
        st.caption("✅ Đoạn truyện ngắn")
        st.caption("✅ Nội dung bài học")
        st.caption("✅ Lời ru, đồng dao")

    btn_convert = st.button(
        "🎵 Chuyển thành bài hát",
        use_container_width=True,
        disabled=not bool((poem_input or "").strip()),
        key="btn_convert_poem"
    )

    if btn_convert and poem_input.strip():
        style_key_poem = poem_style.split("–")[0].strip()
        style_for_poem = STYLE_MAP.get(style_key_poem, DEFAULT_SUNOSTYLE)
        try:
            with st.spinner("Đang chuyển đổi nội dung thành lời bài hát..."):
                converted = poem_to_song(poem_input, poem_age, poem_style)
            st.session_state.lyrics = converted
            st.session_state.title = poem_title or "Bài Hát Thiếu Nhi"
            st.session_state.topic = poem_title or "Chuyển từ thơ/truyện"
            st.session_state["poem_style_for_api"] = style_for_poem
            st.session_state["poem_converted"] = True
            st.success("✅ Đã chuyển đổi thành công! Xem kết quả bên dưới.")
        except Exception as e:
            st.error(f"Lỗi: {e}")

    if st.session_state.get("poem_converted") and st.session_state.lyrics:
        st.markdown("---")
        st.markdown("#### 📄 Lời bài hát sau khi chuyển đổi")
        st.session_state.lyrics = st.text_area(
            "Chỉnh sửa lời nếu cần:",
            value=st.session_state.lyrics,
            height=300,
            key="poem_lyrics_edit"
        )

        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            poem_instrumental = st.toggle("Chỉ giai điệu (không lời)", value=False, key="poem_instrumental")
        with col_r2:
            btn_poem_music = st.button("🎧 Tạo nhạc từ lời này", use_container_width=True, key="btn_poem_music")

        if btn_poem_music:
            style_for_api_poem = st.session_state.get("poem_style_for_api", DEFAULT_SUNOSTYLE)
            try:
                with st.spinner("Đang tạo bài hát..."):
                    task_id = suno_generate_song(
                        st.session_state.lyrics,
                        st.session_state.title or "Kids Song",
                        style=style_for_api_poem,
                        instrumental=poem_instrumental,
                    )
                    tracks = suno_poll(task_id)

                st.subheader("🎧 Kết quả")
                ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                base = ascii_slugify(st.session_state.title or "Kids_Song")

                for i, t in enumerate(tracks, 1):
                    audio_url_orig = t.get("audioUrlHigh") or t.get("audioUrl") or ""
                    mp3_path = ""
                    audio_bytes = b""

                    if audio_url_orig:
                        audio_bytes = download_bytes(audio_url_orig)
                        mp3_path = os.path.join(MP3_DIR, f"{ts}_{i}_{base}.mp3")
                        with open(mp3_path, "wb") as f:
                            f.write(audio_bytes)

                    audio_url_pub = None
                    if audio_bytes:
                        audio_url_pub = sb_upload_bytes(
                            SUPABASE_BUCKET,
                            f"mp3/{ts}_{i}_{base}.mp3",
                            audio_bytes,
                            "audio/mpeg",
                        )
                    audio_url_final = audio_url_pub or audio_url_orig or ""

                    st.write(f"**{st.session_state.title} — Bản {i}**")
                    if audio_url_pub:
                        st.caption("☁️ Đã lưu lên Supabase")
                    if mp3_path and os.path.exists(mp3_path):
                        with open(mp3_path, "rb") as f:
                            mp3_data = f.read()
                        st.audio(mp3_data, format="audio/mp3")
                        st.download_button(
                            "⬇️ Tải MP3",
                            data=mp3_data,
                            file_name=os.path.basename(mp3_path),
                            mime="audio/mpeg",
                            use_container_width=True,
                            key=f"dl_poem_{ts}_{i}",
                        )
                    elif audio_url_final:
                        st.audio(audio_url_final, format="audio/mp3")

                    row = {
                        "time": ts,
                        "title": st.session_state.title or "Kids Song",
                        "topic": poem_title or "Chuyển từ thơ/truyện",
                        "keywords": "",
                        "style": poem_style,
                        "language": "vi",
                        "verses": "",
                        "bridge": True,
                        "instrumental": poem_instrumental,
                        "track_index": i,
                        "audio_url": audio_url_final,
                        "image_url": "",
                        "mp3_path": mp3_path if not audio_url_pub else "",
                        "cover_path": "",
                        "lyrics": st.session_state.lyrics,
                        "age_group": poem_age,
                        "theme_month": "",
                        "source_type": "poem",
                    }
                    write_history_row(row)
                    sb_save_track_metadata(row)

                st.balloons()
                st.success("✅ Đã lưu bài hát! Xem ở tab 📚 Thư viện.")
                st.session_state["poem_converted"] = False
            except Exception as e:
                st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)

# ================== TAB 2: THƯ VIỆN ==================
with tab_library:
    if "lib_page" not in st.session_state:
        st.session_state.lib_page = 1

    def reset_page():
        st.session_state.lib_page = 1

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📚 Thư viện (Gallery)")

    df = load_history_df_local()
    data_source = "local"

    # Ưu tiên tải từ Supabase nếu đã kết nối — tránh mất dữ liệu khi app restart trên cloud
    if supabase:
        df_sb = load_history_from_supabase()
        if df_sb is not None and len(df_sb) > 0:
            df = df_sb
            data_source = "Supabase ☁️"

    if df is None or len(df) == 0:
        st.info("Chưa có bài nhạc nào trong thư viện.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        colf1, colf2, colf3, colf4 = st.columns([1.2, 1, 1, 1])
        with colf1:
            q = st.text_input("Tìm theo tiêu đề/chủ đề", key="lib_kw", on_change=reset_page).strip()
        with colf2:
            style_vals = sorted(df["style"].dropna().unique().tolist()) if "style" in df.columns else []
            style_pick = st.selectbox(
                "Lọc theo phong cách",
                ["Tất cả"] + style_vals,
                index=0,
                on_change=reset_page,
                key="lib_style",
            )
        with colf3:
            age_vals = sorted(df["age_group"].dropna().unique().tolist()) if "age_group" in df.columns else []
            age_pick = st.selectbox(
                "Lọc theo độ tuổi",
                ["Tất cả"] + age_vals,
                index=0,
                on_change=reset_page,
                key="lib_age",
            )
        with colf4:
            sort_opt = st.selectbox(
                "Sắp xếp",
                ["Mới nhất", "A→Z", "Theo style"],
                index=0,
                on_change=reset_page,
                key="lib_sort",
            )

        if q:
            qn = norm_txt(q)
            mask = pd.Series(False, index=df.index)
            for col in ["title", "topic"]:
                if col in df.columns:
                    mask = mask | df[col].astype(str).map(norm_txt).str.contains(qn, na=False)
            df = df[mask]

        if style_pick != "Tất cả" and "style" in df.columns:
            df = df[df["style"] == style_pick]

        if age_pick != "Tất cả" and "age_group" in df.columns:
            df = df[df["age_group"] == age_pick]

        if "time" in df.columns:
            df["time_dt"] = df["time"].apply(parse_time_safe)

        if sort_opt == "Mới nhất" and "time_dt" in df.columns:
            df = df.sort_values("time_dt", ascending=False, na_position="last")
        elif sort_opt == "A→Z" and "title" in df.columns:
            df = df.sort_values("title", key=lambda s: s.astype(str).str.lower(), ascending=True)
        elif sort_opt == "Theo style" and "style" in df.columns:
            secondary = "title" if "title" in df.columns else "style"
            df = df.sort_values(["style", secondary], ascending=[True, True])

        st.caption(f"🔎 Nguồn dữ liệu: **{data_source}**")

        if len(df) == 0:
            st.info("Không có bài nào khớp bộ lọc.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            cv1, cv2 = st.columns([1, 1])
            with cv1:
                layout_mode = st.radio("Kiểu hiển thị", ["Danh sách", "Lưới"], horizontal=True, index=0)
            with cv2:
                page_size = st.selectbox("Số bài hát/trang", [8, 12, 16, 24], index=0)

            total = len(df)
            page_count = max(1, (total + page_size - 1) // page_size)
            colp1, colp2 = st.columns([1.5, 1])
            with colp1:
                start_show = min((st.session_state.lib_page - 1) * page_size + 1, total)
                end_show = min(st.session_state.lib_page * page_size, total)
                st.caption(f"Hiển thị {start_show}–{end_show} / {total}")
            with colp2:
                st.session_state.lib_page = st.number_input(
                    "Trang",
                    min_value=1,
                    max_value=page_count,
                    value=min(st.session_state.lib_page, page_count),
                    step=1,
                    key="lib_page_input",
                )

            start = (st.session_state.lib_page - 1) * page_size
            end = start + page_size
            df_page = df.iloc[start:end].reset_index(drop=True)

            NCOLS = 4
            if layout_mode == "Lưới":
                cols = st.columns(NCOLS)
                for idx, row in df_page.iterrows():
                    with cols[idx % NCOLS]:
                        st.markdown('<div class="card-sm">', unsafe_allow_html=True)
                        show_cover_from_row(row)
                        st.markdown(f"**{row.get('title') or 'Kids Song'}**")
                        subtitle = str(row.get("time", ""))
                        if str(row.get("track_index", "")).strip():
                            subtitle += f" · Bản {row.get('track_index')}"
                        st.caption(subtitle)
                        show_audio_from_row(row, key_suffix=f"grid_{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                for idx, row in df_page.iterrows():
                    col_img, col_meta = st.columns([1, 2.4])
                    with col_img:
                        show_cover_from_row(row)
                    with col_meta:
                        st.markdown('<div class="card-sm">', unsafe_allow_html=True)
                        st.markdown(f"**{row.get('title') or 'Kids Song'}**")
                        subtitle = str(row.get("time", ""))
                        if str(row.get("track_index", "")).strip():
                            subtitle += f" · Bản {row.get('track_index')}"
                        st.caption(subtitle)
                        st.write(f"**Chủ đề:** {row.get('topic', '')}")
                        st.write(f"**Phong cách:** {row.get('style', '')}")
                        show_audio_from_row(row, key_suffix=f"list_{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.divider()

            if st.button("🔄 Làm mới thư viện", key="btn_refresh_library", use_container_width=True):
                if hasattr(st, "cache_data"):
                    st.cache_data.clear()
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

# ================== TAB 4: THỐNG KÊ ==================
with tab_stats:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Thống kê sử dụng")

    df_stats = load_history_df_local()
    if supabase:
        df_sb_stats = load_history_from_supabase()
        if df_sb_stats is not None and len(df_sb_stats) > 0:
            df_stats = df_sb_stats

    if df_stats is None or df_stats.empty:
        st.info("Chưa có dữ liệu thống kê. Hãy tạo một vài bài hát trước!")
    else:
        # Tổng quan
        total = len(df_stats)
        tong_tho = len(df_stats[df_stats.get("source_type", pd.Series()) == "poem"]) if "source_type" in df_stats.columns else 0
        tong_moi = total - tong_tho

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎵 Tổng bài hát", total)
        c2.metric("✨ Tạo mới", tong_moi)
        c3.metric("📖 Từ thơ/truyện", tong_tho)
        styles_count = df_stats["style"].nunique() if "style" in df_stats.columns else 0
        c4.metric("🎼 Phong cách", styles_count)

        st.divider()

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### 🏆 Chủ đề được tạo nhiều nhất")
            if "topic" in df_stats.columns:
                topic_counts = df_stats["topic"].value_counts().head(8)
                for i, (topic_name, count) in enumerate(topic_counts.items(), 1):
                    st.write(f"{i}. **{topic_name}** — {count} bài")

        with col_s2:
            st.markdown("#### 🎼 Phong cách nhạc phổ biến")
            if "style" in df_stats.columns:
                style_counts = df_stats["style"].value_counts().head(6)
                for style_name, count in style_counts.items():
                    bar = "█" * min(count * 2, 20)
                    st.write(f"**{str(style_name)[:30]}** {bar} {count}")

        st.divider()

        col_s3, col_s4 = st.columns(2)

        with col_s3:
            st.markdown("#### 👶 Phân loại theo độ tuổi")
            if "age_group" in df_stats.columns:
                age_counts = df_stats["age_group"].value_counts()
                for age, count in age_counts.items():
                    if str(age).strip():
                        st.write(f"- **{age}**: {count} bài")
            else:
                st.caption("Chưa có dữ liệu độ tuổi (cần tạo bài hát mới)")

        with col_s4:
            st.markdown("#### 📅 Chủ đề theo tháng")
            if "theme_month" in df_stats.columns:
                theme_counts = df_stats[df_stats["theme_month"].astype(str).str.strip() != ""]["theme_month"].value_counts()
                if len(theme_counts) > 0:
                    for theme, count in theme_counts.items():
                        st.write(f"- {theme}: **{count} bài**")
                else:
                    st.caption("Chưa có bài hát theo chủ đề tháng")
            else:
                st.caption("Chưa có dữ liệu chủ đề tháng")

        st.divider()
        st.markdown("#### 📋 Số bài hát theo ngày tạo")
        if "time" in df_stats.columns:
            df_stats["date"] = df_stats["time"].astype(str).str[:8]
            date_counts = df_stats["date"].value_counts().sort_index()
            if len(date_counts) > 0:
                for date_str, count in date_counts.tail(10).items():
                    try:
                        d = dt.datetime.strptime(date_str, "%Y%m%d").strftime("%d/%m/%Y")
                    except Exception:
                        d = date_str
                    bar = "▓" * min(count * 3, 30)
                    st.write(f"`{d}` {bar} {count} bài")

    st.markdown('</div>', unsafe_allow_html=True)

# ================== TAB 3: LỊCH SỬ ==================
with tab_history:
    st.subheader("🗂️ Đây là nơi giáo viên có thể tìm lại các nguồn bài hát đã tạo ra theo danh sách")

    try:
        df_all = load_history_df_local()
    except Exception as e:
        st.warning(f"Lỗi khi tải lịch sử: {e}")
        df_all = pd.DataFrame(columns=EXPECTED_HEADER)

    if df_all is None or df_all.empty:
        st.info("Chưa có dữ liệu lịch sử nào được ghi nhận.")
    else:
        rename_cols = {
            "time": "Thời gian",
            "title": "Tên bài hát",
            "topic": "Chủ đề / Gợi ý",
            "style": "Phong cách nhạc",
            "audio_url": "Liên kết nhạc",
            "track_index": "Bản số",
            "language": "Ngôn ngữ",
        }
        for k in rename_cols.keys():
            if k not in df_all.columns:
                df_all[k] = ""
        df_all = df_all.rename(columns=rename_cols)

        df_all["Thời gian_dt"] = df_all["Thời gian"].apply(parse_time_safe)
        df_all = df_all.sort_values("Thời gian_dt", ascending=False, na_position="last").reset_index(drop=True)

        st.markdown("### 🔍 Bộ lọc dữ liệu")
        col1, col2 = st.columns(2)

        with col1:
            valid_times = df_all["Thời gian_dt"].dropna()
            date_range = None
            if len(valid_times) > 0:
                min_date = valid_times.min().date()
                max_date = valid_times.max().date()
                date_range = st.date_input(
                    "Chọn khoảng thời gian",
                    (min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
            else:
                st.caption("Không xác định được thời gian cho các bản ghi cũ — hiển thị toàn bộ.")

        with col2:
            style_options = ["Tất cả"] + sorted(
                [s for s in df_all["Phong cách nhạc"].dropna().astype(str).unique() if s.strip()]
            )
            style_filter = st.selectbox("Chọn phong cách nhạc", style_options, index=0)

        df_filtered = df_all.copy()
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
            start_dt = dt.datetime.combine(start, dt.time.min)
            end_dt = dt.datetime.combine(end, dt.time.max)
            has_time = df_filtered["Thời gian_dt"].notna()
            in_range = (df_filtered["Thời gian_dt"] >= start_dt) & (df_filtered["Thời gian_dt"] <= end_dt)
            df_filtered = pd.concat(
                [df_filtered[has_time & in_range], df_filtered[~has_time]],
                ignore_index=True,
            )

        if style_filter != "Tất cả":
            df_filtered = df_filtered[df_filtered["Phong cách nhạc"].astype(str) == style_filter]

        st.success(f"Hiển thị {len(df_filtered)} bản ghi sau khi lọc.")

        def fmt_time(row):
            v = row.get("Thời gian_dt")
            if pd.isna(v):
                return str(row.get("Thời gian") or "")
            try:
                return v.strftime("%d-%m-%Y %H:%M:%S")
            except Exception:
                return str(row.get("Thời gian") or "")

        df_show = df_filtered.copy()
        df_show["Thời gian"] = df_show.apply(fmt_time, axis=1)

        st.markdown("### 📄 Danh sách lịch sử chi tiết")
        cols_show = [
            "Thời gian",
            "Tên bài hát",
            "Chủ đề / Gợi ý",
            "Phong cách nhạc",
            "Ngôn ngữ",
            "Bản số",
            "Liên kết nhạc",
        ]
        cols_show = [c for c in cols_show if c in df_show.columns]
        st.dataframe(df_show[cols_show], use_container_width=True, height=420)

        st.markdown("### ⬇️ Tải dữ liệu xuất báo cáo")
        csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Tải về CSV",
            data=csv_bytes,
            file_name=f"lich_su_tao_nhac_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        try:
            import xlsxwriter  # noqa: F401

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df_show.to_excel(writer, index=False, sheet_name="LichSu")
            st.download_button(
                "Tải về Excel",
                data=buf.getvalue(),
                file_name=f"lich_su_tao_nhac_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            st.caption("Cài thêm xlsxwriter nếu muốn xuất Excel.")

# ================= TAB 4: CÀI ĐẶT =================
with tab_settings:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if supabase:
        st.success("Đã kết nối Supabase.")
        btn_list = st.button("🔎 Xem file mp3 trong bucket")
        if btn_list:
            try:
                mp3_files = supabase.storage.from_(SUPABASE_BUCKET).list("mp3") or []
                cover_files = supabase.storage.from_(SUPABASE_BUCKET).list("covers") or []
                st.write(f"📁 **mp3/**: {len(mp3_files)} file | **covers/**: {len(cover_files)} file")
                for f in mp3_files[:20]:
                    name = f.get("name")
                    pub = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(f"mp3/{name}")
                    url = pub.get("publicUrl") if isinstance(pub, dict) else str(pub)
                    st.markdown(f"- 🎵 `{name}` → [nghe]({url})")
                for f in cover_files[:10]:
                    name = f.get("name")
                    pub = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(f"covers/{name}")
                    url = pub.get("publicUrl") if isinstance(pub, dict) else str(pub)
                    st.markdown(f"- 🖼️ `{name}` → [xem]({url})")
            except Exception as e:
                st.warning(f"Lỗi đọc bucket: {e}")
    else:
        st.info("Chưa kết nối Supabase. App vẫn lưu local bình thường tại outputs/mp3 và outputs/covers.")

    st.divider()
    st.markdown("### 🎨 Preset chủ đề nhanh")
    preset = st.selectbox(
        "Chọn nhanh",
        [
            "Màu sắc cơ bản",
            "Hình tròn – vuông – tam giác",
            "Số đếm 1 – 10",
            "Vệ sinh răng miệng",
            "Chào hỏi & phép lịch sự",
            "An toàn giao thông",
            "Con vật",
            "Gia đình",
            "Nghề nghiệp",
            "Trường mầm non",
            "Bản thân bé",
            "Thầy cô và bạn bè",
        ],
    )
    st.caption(f"Chủ đề gợi ý: {preset}")

    st.divider()
    st.markdown("### ℹ️ Ghi chú")
    st.markdown(
        "- **Refine** chỉnh câu từ tùy ý, không đổi chủ đề.\n"
        "- **Instrumental** chỉ tạo giai điệu không lời.\n"
        "- Ảnh bìa lỗi vẫn không làm mất bài hát.\n"
        "- Thư viện và lịch sử hiện cùng đọc một file: `outputs/tracks.csv`."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown(
    """
<hr style="margin:24px 0; border:none; border-top:1px solid #e6e8f5;">
<div style="text-align:center; margin-top:8px; line-height:1.7;">
  <div style="font-weight:800; font-size:18px;">© NHẠC AI THIẾU NHI • Dành cho Giáo viên mầm non</div>
  <div style="font-size:15px; color:#64748b;"> Facebook: Ngọc Thảo – <a href=\"mailto:ms.nthaotran@gmail.com\">ms.nthaotran@gmail.com</a></div>
</div>
""",
    unsafe_allow_html=True,
)
