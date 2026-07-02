import os, time, json, requests, datetime as dt, csv, re, io, base64, unicodedata
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_secret(name, default=None):
    try: return st.secrets.get(name, os.getenv(name, default))
    except: return os.getenv(name, default)

OPENAI_API_KEY       = get_secret("OPENAI_API_KEY")
SUNO_API_KEY         = get_secret("SUNO_API_KEY")
SUNO_API_BASE        = get_secret("SUNO_API_BASE", "https://api.sunoapi.org")
SUNO_MODEL           = get_secret("SUNO_MODEL", "V5")
SUNO_CALLBACK_URL    = get_secret("SUNO_CALLBACK_URL")
DEFAULT_SUNOSTYLE    = get_secret("DEFAULT_SUNOSTYLE", "Children's Pop, cheerful, simple melody, clapping beat, preschool music")

R2_ACCOUNT_ID        = get_secret("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID     = get_secret("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = get_secret("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME       = get_secret("R2_BUCKET_NAME", "kids-songs")
R2_PUBLIC_URL        = get_secret("R2_PUBLIC_URL", "")

ANALYTICS_CSV = get_secret("ANALYTICS_CSV", "du_lieu_tao_nhac.csv")

if not OPENAI_API_KEY: st.error("Thiếu OPENAI_API_KEY"); st.stop()
if not SUNO_API_KEY:   st.error("Thiếu SUNO_API_KEY"); st.stop()
if not SUNO_CALLBACK_URL: st.warning("Chưa có SUNO_CALLBACK_URL — app vẫn hoạt động bằng cách poll kết quả.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client  = OpenAI(api_key=OPENAI_API_KEY)
HEADERS = {"Authorization": f"Bearer {SUNO_API_KEY}", "Content-Type": "application/json"}

r2_client, r2_status = None, "❌"
if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    try:
        r2_client = boto3.client("s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto")
        r2_client.head_bucket(Bucket=R2_BUCKET_NAME)
        r2_status = "✅"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        st.warning(f"Bucket '{R2_BUCKET_NAME}' chưa tồn tại." if code == "404" else f"Lỗi R2: {e}")
        r2_client = None
    except Exception as e:
        st.warning(f"Không kết nối được R2: {e}"); r2_client = None

OUTPUT_DIR = "outputs"; MP3_DIR = os.path.join(OUTPUT_DIR,"mp3"); COVER_DIR = os.path.join(OUTPUT_DIR,"covers")
IMG_DIR = os.path.join(OUTPUT_DIR,"illustrations")
os.makedirs(MP3_DIR, exist_ok=True); os.makedirs(COVER_DIR, exist_ok=True); os.makedirs(IMG_DIR, exist_ok=True)

HISTORY_CSV = os.path.join(OUTPUT_DIR, "tracks.csv")
EXPECTED_HEADER = ["time","title","topic","keywords","style","language","verses","bridge",
    "instrumental","track_index","audio_url","image_url","mp3_path","cover_path",
    "lyrics","age_group","theme_month","source_type",
    "content_type","content_text","scenes","teaching_guide","skill_goal"]

# ── Chủ đề kỹ năng / thái độ / học tập ──
CONTENT_THEMES = {
    "🧼 Kỹ năng sống": {
        "Rửa tay sạch": "Kỹ năng rửa tay đúng cách, phòng bệnh, vệ sinh cá nhân",
        "Đánh răng buổi sáng": "Thói quen đánh răng, chăm sóc răng miệng hàng ngày",
        "Mặc quần áo gọn gàng": "Tự mặc quần áo, gấp đồ ngăn nắp, tự lập",
        "Chào hỏi lễ phép": "Chào hỏi người lớn, lễ phép, văn hóa giao tiếp",
        "Ăn uống gọn gàng": "Ngồi ăn ngay ngắn, không rơi vãi, biết ơn bữa ăn",
        "Dọn dẹp đồ chơi": "Tự dọn dẹp sau khi chơi, gọn gàng ngăn nắp",
        "Vượt qua sợ hãi": "Dũng cảm thử điều mới, không sợ té ngã, tự tin",
    },
    "❤️ Thái độ & Cảm xúc": {
        "Yêu thương bạn bè": "Quan tâm, chia sẻ, giúp đỡ bạn bè trong lớp",
        "Chia sẻ đồ chơi": "Biết nhường nhịn, chia sẻ, không tranh giành",
        "Xin lỗi và tha thứ": "Nhận lỗi, nói xin lỗi, tha thứ cho người khác",
        "Cảm ơn và biết ơn": "Nói cảm ơn, trân trọng sự giúp đỡ, lòng biết ơn",
        "Kiên nhẫn chờ đợi": "Biết chờ đợi đến lượt, không nóng vội, kiên nhẫn",
        "Vui vẻ mỗi ngày": "Nụ cười, lạc quan, tìm điều vui trong cuộc sống",
        "Dũng cảm và tự tin": "Mạnh dạn phát biểu, thử thách mới, tin vào bản thân",
    },
    "🧠 Học tập & Nhận thức": {
        "Màu sắc cơ bản": "Học nhận biết màu đỏ, xanh, vàng, tím, cam, hồng",
        "Số đếm 1-10": "Học đếm số từ 1 đến 10, nhận biết chữ số",
        "Hình dạng cơ bản": "Hình tròn, vuông, tam giác, chữ nhật, ngôi sao",
        "Chữ cái tiếng Việt": "Học bảng chữ cái, nhận biết chữ, bắt đầu tập đọc",
        "Các mùa trong năm": "Xuân hạ thu đông, đặc điểm từng mùa, thời tiết",
        "Ngày và đêm": "Mặt trời, mặt trăng, ngày đêm, buổi sáng tối",
        "Con số và phép đếm": "Nhiều ít, to nhỏ, cao thấp, so sánh số lượng",
    },
}

# ── Chủ đề tháng GDMN ──
CHU_DE_THANG = {
    "Tháng 9 — Trường mầm non":{"mo_ta":"Trường mầm non thân yêu của bé","tu_khoa":"Trường lớp, cô giáo, bạn bè, sân chơi, đồ chơi"},
    "Tháng 10 — Bản thân":{"mo_ta":"Cơ thể bé và những điều bé thích","tu_khoa":"Tay chân, mặt mũi, tên bé, sở thích, cảm xúc"},
    "Tháng 11 — Gia đình":{"mo_ta":"Gia đình yêu thương của bé","tu_khoa":"Ba mẹ, ông bà, anh chị em, ngôi nhà, yêu thương"},
    "Tháng 12 — Nghề nghiệp":{"mo_ta":"Các nghề nghiệp trong xã hội","tu_khoa":"Bác sĩ, cô giáo, chú công an, nông dân, kỹ sư"},
    "Tháng 1-2 — Thế giới động vật":{"mo_ta":"Các con vật gần gũi với bé","tu_khoa":"Chó mèo, gà vịt, bướm sâu, rừng núi, biển cả"},
    "Tháng 3 — Thế giới thực vật":{"mo_ta":"Cây cối hoa lá xung quanh bé","tu_khoa":"Hoa lá, cây xanh, rau củ, mùa xuân, vườn cây"},
    "Tháng 3-4 — Phương tiện giao thông":{"mo_ta":"Các phương tiện giao thông bé biết","tu_khoa":"Xe ô tô, xe đạp, máy bay, tàu thuyền, an toàn"},
    "Tháng 4-5 — Quê hương Đất nước":{"mo_ta":"Tình yêu quê hương đất nước Việt Nam","tu_khoa":"Quê hương, Việt Nam, cờ đỏ, sao vàng, biển đảo"},
    "Tháng 5-6 — Mùa hè":{"mo_ta":"Mùa hè vui tươi và kỳ nghỉ hè","tu_khoa":"Nắng vàng, biển xanh, kem mát, bướm hoa, nghỉ hè"},
}

AGE_GROUPS = ["Nhà trẻ (0-3 tuổi)","Mẫu giáo bé (3-4 tuổi)","Mẫu giáo nhỡ (4-5 tuổi)","Mẫu giáo lớn (5-6 tuổi)"]
LINH_VUC   = ["🏃 Phát triển thể chất","🧠 Phát triển nhận thức","🗣️ Phát triển ngôn ngữ","🎨 Phát triển thẩm mỹ","❤️ Phát triển tình cảm - Xã hội"]
STYLE_DISPLAY = ["Children's Pop – Nhạc pop thiếu nhi, tươi vui, dễ hát","Playful / Upbeat Kids – Nhạc vui nhộn, hoạt bát","Nursery Rhymes – Đồng dao, hát thiếu nhi cổ điển","Educational Songs – Nhạc học tập, dạy chữ, đếm số","Children's Folk – Dân ca thiếu nhi, nhẹ nhàng","Lullaby – Nhạc ru, dễ ngủ, êm dịu","Magical / Whimsical Kids – Huyền ảo, cổ tích","Children's Jazz – Nhạc jazz nhẹ, thư giãn","Musical Story / Narrative – Nhạc kể chuyện","Children's Rock","Children's Rap"]
STYLE_MAP = {
    "Children's Pop":"Vietnamese children's pop for preschool. Melody bright, simple. Tempo medium-upbeat, clapping. Instruments: guitar, xylophone. Vocals: warm child voice. Mood: cheerful, safe for 3-6.",
    "Playful / Upbeat Kids":"Upbeat playful kids music Vietnamese preschool. Tempo fast bouncy. Xylophone, ukulele, hand claps. Energetic child voice. Mood: very happy, active.",
    "Nursery Rhymes":"Vietnamese nursery rhyme. Melody simple, repetitive. Tempo slow-medium. Soft piano, gentle bells. Warm cozy bedtime feeling.",
    "Educational Songs":"Vietnamese educational preschool song. Call-and-response format. Simple melody, keyboard, xylophone. Clear teacher-like voice. Mood: learning, engaging.",
    "Children's Folk":"Vietnamese children's folk (dân ca thiếu nhi). Pentatonic scale. Sáo trúc, đàn tranh, light percussion. Gentle child voice, folk style. Mood: nostalgic, countryside.",
    "Lullaby":"Vietnamese lullaby. Very slow 60-70 BPM. Soft guitar or đàn tranh. Tender motherly voice. Mood: peaceful, sleepy, loving.",
    "Magical / Whimsical Kids":"Magical children's song. Dreamy melody. Bells, glockenspiel, soft piano. Wonder-filled child voice. Mood: magical, imaginative.",
    "Children's Jazz":"Soft jazz for Vietnamese preschool. Swinging melody. Soft piano, brushed drums, light bass. Fun yet child-friendly.",
    "Musical Story / Narrative":"Vietnamese children's musical story. Dynamic melody. Strings, piano, light orchestra. Expressive storytelling voice.",
    "Children's Rock":"Vietnamese children's rock, simple playful. Medium tempo 90-110 BPM. Clean guitar, bass, simple drums. Happy energetic safe for preschool.",
    "Children's Rap":"Vietnamese children's rap chant. Slow flow, spoken-sung. Light hip-hop beat, soft kick, hand clap. Clear Vietnamese. Fun educational.",
}

DEFAULT_LYRICS_SYSTEM = (
    "Bạn là chuyên gia sáng tác nhạc thiếu nhi và nhà sư phạm mầm non giàu kinh nghiệm. "
    "Sáng tác lời bài hát cho trẻ 0-6 tuổi. Đối với trẻ lứa tuổi 0-3 tuổi, mỗi câu hát từ 5 từ, đoạn ngắn, vần điệu đơn giản. Đối với trẻ lứa tuổi 3-6 tuổi, mỗi câu hát 5-10 từ. "
    "Ngôn ngữ giáo dục, nhân văn. Vần điệu gieo vần, rõ ràng, điệp khúc dễ nhớ. "
    "Mặc định dùng [Verse] và [Chorus]. CHỈ dùng cấu trúc rap khi yêu cầu rõ.")

# ════════════════════════════════════════════════════
# HÀM AI
# ════════════════════════════════════════════════════

def generate_lyrics(topic, target_words=None, language="vi", verses=2, bridge=True):
    tw = ", ".join(target_words) if target_words else "Không bắt buộc"
    structure = ["[Verse 1] → [Chorus]"] + [f"→ [Verse {i}] → [Chorus]" for i in range(2, verses+1)]
    if bridge: structure.append("→ [Bridge] → [Chorus]")
    prompt = f"Chủ đề: {topic}\nNgôn ngữ: {language}\nCấu trúc: {' '.join(structure)}\nTừ khóa: {tw}\nĐộ dài ~12-18 dòng."
    r = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"system","content":DEFAULT_LYRICS_SYSTEM},{"role":"user","content":prompt}],
        temperature=0.9, max_tokens=700)
    return r.choices[0].message.content.strip()

def refine_lyrics(original_text, instruction=""):
    if not original_text.strip(): return original_text
    r = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"system","content":DEFAULT_LYRICS_SYSTEM},
                  {"role":"user","content":f"Chỉnh sửa lời bài hát thiếu nhi. Chỉ dẫn: {instruction or 'Không có'}\n\n{original_text}"}],
        temperature=0.6, max_tokens=800)
    return r.choices[0].message.content.strip()

def poem_to_song(poem_text, age_group="Mẫu giáo (3-6 tuổi)", style_hint=""):
    r = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"system","content":"Chuyển thơ/truyện thiếu nhi thành lời bài hát. Cấu trúc [Verse]/[Chorus]/[Bridge], vần điệu rõ, câu 5-10 từ trở lên."},
                  {"role":"user","content":f"Lứa tuổi: {age_group}\nPhong cách: {style_hint}\n\n{poem_text}\n\nChuyển thành lời bài hát."}],
        temperature=0.8, max_tokens=800)
    return r.choices[0].message.content.strip()

def generate_poem(topic, age_group, skill_goal, poem_type="tho"):
    """Tạo bài thơ hoặc câu chuyện thiếu nhi có tính giáo dục."""
    if poem_type == "tho":
        system = (
            "Bạn là nhà thơ thiếu nhi Việt Nam giàu kinh nghiệm sư phạm mầm non. "
            "Viết bài thơ ngắn cho trẻ mầm non: đối với lứa tuổi trẻ từ 0-3 tuổi có 2-3 khổ thơ, đối với lứa tuổi trẻ từ 3-5 tuổi có 4-6 khổ thơ, mỗi khổ 4 câu, vần điệu gieo vần rõ ràng, "
            "ngôn ngữ trong sáng, hình ảnh sinh động, dễ thuộc, mang thông điệp giáo dục tích cực. "
            "Không có nội dung tiêu cực, bạo lực. Thêm tiêu đề bài thơ ở đầu."
        )
        user = f"Chủ đề: {topic}\nMục tiêu giáo dục: {skill_goal}\nĐộ tuổi: {age_group}\n\nViết bài thơ thiếu nhi."
    else:
        system = (
            "Bạn là nhà văn chuyên viết truyện thiếu nhi Việt Nam. "
            "Viết câu chuyện ngắn có từ 150-200 từ cho trẻ từ 0-3 tuổi và có 300-400 từ cho trẻ từ 3-6 tuổi: có nhân vật dễ thương (người hoặc bất kỳ sự vật nào có thể nhân hóa được(ví dụ: con vật, cây cối, quả, hiện tượng tự nhiên...), "
            "tình huống gần gũi, kết thúc có thông điệp tích cực rõ ràng. "
            "Chia thành 3-4 đoạn ngắn. Ngôn ngữ đơn giản, hình ảnh sinh động. Thêm tiêu đề ở đầu."
        )
        user = f"Chủ đề: {topic}\nMục tiêu giáo dục: {skill_goal}\nĐộ tuổi: {age_group}\n\nViết câu chuyện thiếu nhi."
    r = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.85, max_tokens=900)
    return r.choices[0].message.content.strip()

def generate_scenes(content, topic, poem_type, age_group):
    """Tạo kịch bản phân cảnh minh họa từ nội dung thơ/truyện."""
    if poem_type == "tho":
        system = (
            "Bạn là chuyên gia sư phạm mầm non và họa sĩ minh họa sách thiếu nhi. "
            "Dựa trên bài thơ, hãy tạo 4-5 PHÂN CẢNH MINH HỌA mô tả chi tiết hình ảnh "
            "mà giáo viên có thể vẽ hoặc in để dạy trẻ. Mỗi phân cảnh gồm: "
            "① Tên cảnh, ② Mô tả hình ảnh chi tiết (nhân vật, màu sắc, bối cảnh), "
            "③ Câu thơ tương ứng, ④ Gợi ý hoạt động cho trẻ với cảnh đó."
        )
        user = f"Bài thơ về '{topic}' cho trẻ {age_group}:\n\n{content}\n\nTạo kịch bản phân cảnh minh họa."
    else:
        system = (
            "Bạn là chuyên gia sư phạm mầm non và họa sĩ minh họa sách thiếu nhi. "
            "Dựa trên câu chuyện, hãy tạo 4-5 PHÂN CẢNH MINH HỌA như một cuốn sách tranh. "
            "Mỗi phân cảnh gồm: ① Tên cảnh/trang, ② Mô tả hình ảnh chi tiết "
            "(nhân vật, biểu cảm, màu sắc, bối cảnh), ③ Đoạn truyện tương ứng, "
            "④ Câu hỏi gợi mở cho trẻ khi xem tranh."
        )
        user = f"Câu chuyện về '{topic}' cho trẻ {age_group}:\n\n{content}\n\nTạo kịch bản phân cảnh minh họa sách tranh."
    r = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.8, max_tokens=1200)
    return r.choices[0].message.content.strip()

def generate_teaching_guide(content, topic, skill_goal, age_group, poem_type):
    """Tạo hướng dẫn sử dụng trong dạy học."""
    ptype = "bài thơ" if poem_type == "tho" else "câu chuyện"
    system = (
        "Bạn là chuyên gia giáo dục mầm non Việt Nam. "
        f"Tạo HƯỚNG DẪN SỬ DỤNG {ptype.upper()} TRONG DẠY HỌC gồm: "
        "① Mục tiêu giáo dục cụ thể (3-4 mục tiêu), "
        "② Cách dẫn dắt trẻ vào bài (2-3 câu hỏi khởi động), "
        "③ Hoạt động trong khi đọc/kể (3-4 hoạt động tương tác), "
        "④ Hoạt động sau khi đọc/kể (2-3 trò chơi/bài tập củng cố), "
        "⑤ Lời nhắn nhủ/thông điệp chốt cho trẻ (1-2 câu ngắn gọn)."
    )
    user = f"Chủ đề: {topic}\nMục tiêu: {skill_goal}\nĐộ tuổi: {age_group}\n\nNội dung:\n{content[:500]}...\n\nTạo hướng dẫn dạy học."
    r = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.7, max_tokens=800)
    return r.choices[0].message.content.strip()



def _suno_err(resp):
    try: msg = resp.json().get("msg") or resp.json().get("message") or resp.text
    except: msg = resp.text
    if resp.status_code == 429: return RuntimeError(f"Suno hết credit: {msg}")
    if resp.status_code == 401: return RuntimeError("SUNO_API_KEY không hợp lệ.")
    return RuntimeError(f"Suno ({resp.status_code}): {msg}")

def suno_generate_song(prompt, title, style, instrumental=False):
    r = requests.post(f"{SUNO_API_BASE}/api/v1/generate", headers=HEADERS, json={
        "prompt": prompt[:1800], "title": title[:64], "style": style[:200],
        "model": SUNO_MODEL, "instrumental": instrumental, "customMode": True, "callBackUrl": SUNO_CALLBACK_URL
    }, timeout=60)
    if r.status_code >= 400: raise _suno_err(r)
    data = r.json()
    if data.get("code") != 200 or not data.get("data",{}).get("taskId"): raise RuntimeError("Suno failed: "+json.dumps(data))
    return data["data"]["taskId"]

def suno_poll(task_id, timeout_sec=360, interval_sec=8):
    started = time.time()
    while time.time()-started < timeout_sec:
        r = requests.get(f"{SUNO_API_BASE}/api/v1/generate/record-info",
            headers={"Authorization":f"Bearer {SUNO_API_KEY}"}, params={"taskId":task_id}, timeout=60)
        if r.status_code >= 400: raise _suno_err(r)
        try:
            items = r.json()["data"]["response"]["sunoData"]
            ready = [it for it in items if it.get("audioUrl") or it.get("audioUrlHigh")]
            if ready: return ready
        except: pass
        time.sleep(interval_sec)
    raise TimeoutError("Hết thời gian chờ kết quả.")

def download_bytes(url):
    r = requests.get(url, timeout=120); r.raise_for_status(); return r.content

def ascii_slugify(text):
    text = unicodedata.normalize("NFKD",(text or "").strip().replace(" ","_"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return (re.sub(r"[^A-Za-z0-9._-]","_",text).strip("._-") or "file")[:80]

def ensure_history_schema():
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(EXPECTED_HEADER)
        return
    with open(HISTORY_CSV,"r",newline="",encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header == EXPECTED_HEADER: return
    rows_old = list(csv.DictReader(open(HISTORY_CSV,encoding="utf-8")))
    tmp = HISTORY_CSV+".tmp"
    with open(tmp,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=EXPECTED_HEADER); w.writeheader()
        for old in rows_old: w.writerow({k:old.get(k,"") for k in EXPECTED_HEADER})
    os.replace(tmp, HISTORY_CSV)

def r2_upload_bytes(object_key, data_bytes, content_type):
    if not r2_client or not data_bytes: return None
    try:
        r2_client.put_object(Bucket=R2_BUCKET_NAME, Key=object_key, Body=data_bytes, ContentType=content_type)
        if R2_PUBLIC_URL: return f"{R2_PUBLIC_URL.rstrip('/')}/{object_key}"
        return r2_client.generate_presigned_url("get_object", Params={"Bucket":R2_BUCKET_NAME,"Key":object_key}, ExpiresIn=604800)
    except Exception as e:
        st.warning(f"Upload R2 thất bại: {e}"); return None

def r2_upload_mp3(key, data): return r2_upload_bytes(key, data, "audio/mpeg")
def r2_upload_image(key, data): return r2_upload_bytes(key, data, "image/png")
def r2_upload_text(key, text): return r2_upload_bytes(key, text.encode("utf-8"), "text/plain; charset=utf-8")

def save_content_to_library(title, topic, skill_goal, age_group, content_type, content_text, scenes="", teaching_guide="", category=""):
    """Luu tho/truyen vao CSV va upload len R2."""
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = ascii_slugify(title or topic)
    content_r2_url = ""
    if r2_client and content_text:
        label = "BAI THO" if content_type == "tho" else "CAU CHUYEN"
        full_doc = f"{label}: {title}\nChu de: {topic} | Muc tieu: {skill_goal} | Do tuoi: {age_group}\n{'='*50}\n\n{content_text}"
        if scenes: full_doc += f"\n\n{'='*50}\nKICH BAN PHAN CANH\n{'='*50}\n{scenes}"
        if teaching_guide: full_doc += f"\n\n{'='*50}\nHUONG DAN DAY HOC\n{'='*50}\n{teaching_guide}"
        content_r2_url = r2_upload_text(f"content/{ts}_{slug}.txt", full_doc) or ""
    row = {
        "time": ts, "title": title or topic, "topic": topic, "keywords": skill_goal,
        "style": category, "language": "vi", "verses": "", "bridge": "", "instrumental": "",
        "track_index": "1", "audio_url": content_r2_url, "image_url": "", "mp3_path": "",
        "cover_path": "", "lyrics": "", "age_group": age_group, "theme_month": "",
        "source_type": "content", "content_type": content_type, "content_text": content_text,
        "scenes": scenes, "teaching_guide": teaching_guide, "skill_goal": skill_goal,
    }
    write_history_row(row)
    return ts, content_r2_url

def r2_list_files(prefix="mp3/"):
    if not r2_client: return []
    try: return r2_client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix).get("Contents", [])
    except: return []

def r2_get_public_url(object_key):
    if R2_PUBLIC_URL: return f"{R2_PUBLIC_URL.rstrip('/')}/{object_key}"
    if r2_client:
        try: return r2_client.generate_presigned_url("get_object", Params={"Bucket":R2_BUCKET_NAME,"Key":object_key}, ExpiresIn=604800)
        except: pass
    return ""

def write_history_row(row):
    ensure_history_schema()
    with open(HISTORY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.DictWriter(f,fieldnames=EXPECTED_HEADER).writerow({k:row.get(k,"") for k in EXPECTED_HEADER})

def load_history_df_local():
    ensure_history_schema()
    try: return pd.read_csv(HISTORY_CSV, dtype=str, keep_default_na=False)
    except: return pd.DataFrame(columns=EXPECTED_HEADER)

def log_prompt_to_csv(row, csv_path=ANALYTICS_CSV):
    cols = ["Thời gian","Tên bài hát","Miêu tả bài hát","Từ ngữ gợi ý","Chủ đề / Gợi ý","Phong cách nhạc","Ngôn ngữ","Số verse","Bridge","Liên kết nhạc","Liên kết lời hát"]
    mapped = {"Thời gian":row.get("time",""),"Tên bài hát":row.get("title",""),"Miêu tả bài hát":row.get("topic",""),
              "Từ ngữ gợi ý":row.get("keywords",""),"Chủ đề / Gợi ý":row.get("topic",""),"Phong cách nhạc":row.get("style",""),
              "Ngôn ngữ":row.get("language",""),"Số verse":row.get("verses",""),"Bridge":row.get("bridge",""),
              "Liên kết nhạc":row.get("audio_url",""),"Liên kết lời hát":row.get("lyrics_url","")}
    base = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in base.columns: base[c] = None
    base["_key"] = base["Tên bài hát"].fillna("").astype(str).str.strip()+"||"+base["Liên kết nhạc"].fillna("").astype(str).str.strip()
    new_key = f"{(mapped['Tên bài hát'] or '').strip()}||{(mapped['Liên kết nhạc'] or '').strip()}"
    if new_key in set(base["_key"]):
        idx = base.index[base["_key"]==new_key][0]
        for c in cols:
            if str(mapped.get(c,"")).strip(): base.at[idx,c] = mapped[c]
    else:
        row_df = pd.DataFrame([[mapped.get(c,"") for c in cols]],columns=cols); row_df["_key"]=new_key
        base = pd.concat([base,row_df],ignore_index=True)
    base.drop(columns=["_key"],errors="ignore").to_csv(csv_path,index=False,encoding="utf-8-sig")

def strip_accents(s): return "".join(ch for ch in unicodedata.normalize("NFD",str(s)) if unicodedata.category(ch)!="Mn")
def norm_txt(s): return strip_accents(str(s).lower().strip())

def parse_time_safe(x):
    s = "" if x is None else str(x).strip()
    for fmt in ["%Y%m%d-%H%M%S","%Y-%m-%d %H:%M:%S","%Y/%m/%d %H:%M:%S","%d/%m/%Y %H:%M:%S","%Y-%m-%d","%d/%m/%Y"]:
        try: return dt.datetime.strptime(s, fmt)
        except: pass
    try: return pd.to_datetime(s, errors="coerce")
    except: return pd.NaT

def show_cover_from_row(row):
    cp = str(row.get("cover_path","") or "").strip(); iu = str(row.get("image_url","") or "").strip()
    if cp and os.path.exists(cp): st.image(cp, use_container_width=True)
    elif iu: st.image(iu, use_container_width=True)
    else: st.image("https://source.unsplash.com/600x400/?preschool+children+cute+classroom", use_container_width=True)

def show_audio_from_row(row, key_suffix=""):
    mp = str(row.get("mp3_path","") or "").strip(); au = str(row.get("audio_url","") or "").strip()
    if mp and os.path.exists(mp):
        data = open(mp,"rb").read()
        st.audio(data, format="audio/mp3")
        st.download_button("⬇ Tải MP3", data=data, file_name=os.path.basename(mp), mime="audio/mpeg",
            use_container_width=True, key=f"dl_{row.get('time','')}_{row.get('track_index','1')}_{key_suffix}")
    elif au: st.audio(au, format="audio/mp3")

# ════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════
st.set_page_config(page_title="MẦM NON STUDIO", page_icon="🎵", layout="centered")

def add_bg_from_local(image_path, alpha=0.85):
    try:
        encoded = base64.b64encode(open(image_path,"rb").read()).decode()
        st.markdown(f"<style>[data-testid='stAppViewContainer']{{background-image:linear-gradient(rgba(255,255,255,{alpha}),rgba(255,255,255,{alpha})),url('data:image/png;base64,{encoded}');background-size:cover;background-position:center;background-repeat:no-repeat;}}</style>",unsafe_allow_html=True)
    except: pass

add_bg_from_local("music2.jpg", alpha=0.88)

st.markdown("""<style>
:root{--bg:#FFFFFF;--primary:#FFB996;--primary-strong:#FFA97A;--chip:#FFE8D9;--input:#F8FAFD;--text:#2D2D2D;--ring:#FFD7C3;--shadow:0 10px 20px rgba(17,24,39,.07);}
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&display=swap');
html,body,.stApp,[class*="css"]{font-family:'Fredoka',system-ui,sans-serif!important;color:var(--text)!important;}
.main .block-container{background:var(--bg)!important;border-radius:16px!important;padding:2rem!important;box-shadow:var(--shadow);}
section[data-testid="stSidebar"]>div{background:var(--chip)!important;}
.stTextInput input,.stSelectbox div[data-baseweb="select"]>div,.stNumberInput input,.stTextArea textarea{background:var(--input)!important;border:1px solid var(--ring)!important;border-radius:12px!important;}
div.stButton>button,div.stDownloadButton>button{background:var(--primary)!important;color:#fff!important;border-radius:12px!important;font-weight:600!important;border:none!important;box-shadow:0 6px 12px rgba(255,185,150,.35)!important;}
div.stButton>button:hover{background:var(--primary-strong)!important;transform:translateY(-1px);}
.badge{display:inline-flex;align-items:center;gap:.35rem;padding:.35rem .7rem;border-radius:999px;background:var(--chip);font-weight:600;}
.stTabs div[data-baseweb="tab-highlight"],.stTabs div[data-baseweb="tab-border"]{display:none!important;}
.stTabs div[data-baseweb="tab-list"]{gap:.5rem!important;border-bottom:none!important;}
.stTabs button[role="tab"]{background:var(--chip)!important;border:1px solid var(--ring)!important;border-radius:999px!important;padding:.4rem .9rem!important;font-weight:600!important;}
.stTabs button[role="tab"][aria-selected="true"]{background:var(--primary)!important;color:#fff!important;}
.illus-card{border:2px solid var(--ring);border-radius:16px;padding:1rem;background:var(--input);margin-top:1rem;}
</style>""", unsafe_allow_html=True)

for k,v in [("lyrics",""),("title",""),("topic",""),("targets",[]),("generated",False),("keywords",""),("language","vi"),("verses",2),("bridge",True)]:
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.markdown("## 👩‍🏫 Hướng dẫn nhanh")
    st.markdown(
        "**✨ Tạo bài hát:**\n1. Chọn chủ đề → Tự động điền\n2. Chọn độ tuổi, phong cách\n3. Tạo lời → Tạo nhạc → Tải MP3\n\n"
        "**📝 Thơ & Truyện:**\n1. Chọn chủ đề kỹ năng\n2. Chọn loại nội dung\n3. Tạo nội dung + hình minh họa\n4. Có thể chuyển thành nhạc\n\n"
        "**📖 Từ thơ/truyện có sẵn:**\n1. Dán nội dung → Chuyển → Tạo nhạc"
    )
    st.divider()
    st.caption(f"Model Suno: **{SUNO_MODEL}**")
    st.caption(f"Cloudflare R2: **{r2_status}**")

st.title("🎵 MẦM NON STUDIO")
st.markdown(
    '<span class="badge">🏫 Dành riêng cho Giáo viên Mầm non</span>&nbsp;'
    '<span class="badge">✨ Tạo nhạc & Thơ & Truyện bằng AI</span>&nbsp;'
    '<span class="badge">🎨 Hình minh họa AI</span>',
    unsafe_allow_html=True)

tab_make, tab_content, tab_poem, tab_library, tab_history, tab_settings = st.tabs([
    "✨ Tạo bài hát", "📝 Thơ & Truyện", "📖 Thơ/Truyện → Nhạc", "📚 Thư viện", "🗂️ Lịch sử", "⚙️ Cài đặt"
])

# ════════════ TAB 1: TẠO BÀI HÁT ════════════
with tab_make:
    st.markdown("#### 📅 Chọn chủ đề theo chương trình GDMN")
    c_theme, c_auto = st.columns([3,1])
    with c_theme:
        selected_theme = st.selectbox("Chủ đề tháng", ["— Tự nhập —"]+list(CHU_DE_THANG.keys()), key="theme_select")
    with c_auto:
        if st.button("✨ Tự động điền", use_container_width=True) and selected_theme != "— Tự nhập —":
            info = CHU_DE_THANG[selected_theme]
            st.session_state.update({"topic":info["mo_ta"],"keywords":info["tu_khoa"],"title":info["mo_ta"]})
            st.rerun()
    st.divider()
    c1, c2 = st.columns([2,1])
    with c1:
        topic      = st.text_input("Miêu tả bài hát", st.session_state.topic or "Trường mầm non của bé")
        target_str = st.text_input("Từ ngữ gợi ý", st.session_state.keywords or "Đồ chơi, sân trường, lớp học, thân thương")
        title      = st.text_input("Tiêu đề bài hát", st.session_state.title or "Trường mầm non của bé")
    with c2:
        verses    = st.number_input("Số verse", 1, 4, int(st.session_state.verses))
        bridge    = st.toggle("Thêm Bridge", value=bool(st.session_state.bridge))
        language  = st.selectbox("Ngôn ngữ", ["Vi","En"], index=0 if st.session_state.language=="vi" else 1)
        age_group = st.selectbox("Độ tuổi", AGE_GROUPS, index=1, key="age_group_tab1")
        linh_vuc  = st.selectbox("Lĩnh vực", LINH_VUC, index=3, key="linh_vuc_tab1")
    st.session_state.update({"topic":topic,"title":title,"keywords":target_str,
        "language":"vi" if str(language).lower().startswith("v") else "en","verses":int(verses),"bridge":bool(bridge)})

    style_display = st.selectbox("Phong cách nhạc", STYLE_DISPLAY)
    style_for_api = STYLE_MAP.get(style_display.split("–")[0].strip(), DEFAULT_SUNOSTYLE)

    st.divider(); st.subheader("📝 TIẾN TRÌNH SÁNG TÁC")
    c1, c2, c3 = st.columns(3)
    with c1: btn_generate = st.button("✨ Tạo lời bài hát", use_container_width=True)
    with c2: refine_hint  = st.text_input("Chỉ dẫn refine", placeholder="Ví dụ: nhịp nhanh hơn…")
    with c3: btn_refine   = st.button("🪄 Refine", use_container_width=True, disabled=not bool(st.session_state.lyrics.strip()))

    if btn_generate:
        try:
            with st.spinner("Đang sáng tác lời..."):
                targets = [w.strip() for w in target_str.split(",") if w.strip()]
                st.session_state.lyrics = generate_lyrics(topic, targets, st.session_state.language, int(verses), bool(bridge))
            st.success("Đã sinh lời. Chỉnh sửa hoặc bấm Refine.")
        except Exception as e: st.error(str(e))

    if btn_refine and st.session_state.lyrics.strip():
        try:
            with st.spinner("Đang chỉnh sửa..."):
                st.session_state.lyrics = refine_lyrics(st.session_state.lyrics, refine_hint)
            st.success("Đã refine lời bài hát.")
        except Exception as e: st.error(str(e))

    st.session_state.lyrics = st.text_area("Soạn thảo/Chỉnh sửa tại đây:", value=st.session_state.lyrics, height=320)
    new_title = st.text_input("🔤 Tên bài hát", st.session_state.title)
    if new_title.strip(): st.session_state.title = new_title.strip()

    st.divider()
    l, r = st.columns([1,2])
    with l: instrumental = st.toggle("Chỉ giai điệu (instrumental)", value=False)
    with r: btn_music = st.button("🎧 Tạo nhạc", use_container_width=True, disabled=not bool(st.session_state.lyrics.strip()))

    if btn_music and st.session_state.lyrics.strip():
        try:
            with st.spinner("Đang tạo bài hát..."):
                task_id = suno_generate_song(st.session_state.lyrics, st.session_state.title or "Kids Song", style_for_api, instrumental)
                tracks  = suno_poll(task_id)
            st.subheader("🎧 Kết quả")
            ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S"); base = ascii_slugify(st.session_state.title or "Kids_Song")
            for i, t in enumerate(tracks, 1):
                audio_url_orig = t.get("audioUrlHigh") or t.get("audioUrl") or ""; image_url_orig = t.get("imageUrl") or ""
                mp3_path = cover_path = ""; audio_bytes = b""
                if audio_url_orig:
                    audio_bytes = download_bytes(audio_url_orig)
                    mp3_path = os.path.join(MP3_DIR,f"{ts}_{i}_{base}.mp3")
                    open(mp3_path,"wb").write(audio_bytes)
                if image_url_orig:
                    try:
                        img_bytes = download_bytes(image_url_orig)
                        cover_path = os.path.join(COVER_DIR,f"{ts}_{i}_{base}.jpg"); open(cover_path,"wb").write(img_bytes)
                    except: cover_path = ""
                audio_url_r2 = r2_upload_mp3(f"mp3/{ts}_{i}_{base}.mp3", audio_bytes) if audio_bytes and r2_client else None
                audio_url_final = audio_url_r2 or audio_url_orig or ""
                k1, k2 = st.columns([1,2])
                with k1:
                    if cover_path and os.path.exists(cover_path): st.image(cover_path, use_container_width=True)
                    elif image_url_orig: st.image(image_url_orig, use_container_width=True)
                with k2:
                    st.write(f"**{st.session_state.title or 'Kids Song'} — Bản {i}**")
                    if audio_url_r2: st.caption("☁️ Đã lưu lên Cloudflare R2")
                    if mp3_path and os.path.exists(mp3_path):
                        mp3_data = open(mp3_path,"rb").read()
                        st.audio(mp3_data, format="audio/mp3")
                        st.download_button("⬇️ Tải MP3", data=mp3_data, file_name=os.path.basename(mp3_path), mime="audio/mpeg", use_container_width=True, key=f"dl_now_{ts}_{i}")
                    elif audio_url_final: st.audio(audio_url_final, format="audio/mp3")
                row = {"time":ts,"title":st.session_state.title or "Kids Song","topic":st.session_state.topic,"keywords":st.session_state.keywords,
                    "style":style_display,"language":st.session_state.language,"verses":st.session_state.verses,"bridge":st.session_state.bridge,
                    "instrumental":instrumental,"track_index":i,"audio_url":audio_url_final,"image_url":image_url_orig,
                    "mp3_path":mp3_path if not audio_url_r2 else "","cover_path":cover_path,"lyrics":st.session_state.lyrics,
                    "age_group":st.session_state.get("age_group_tab1",""),"theme_month":st.session_state.get("theme_select",""),"source_type":"new"}
                write_history_row(row); log_prompt_to_csv(row)
            st.balloons()
            st.success(f"Đã lưu bài hát{'  và đồng bộ lên **Cloudflare R2** ☁️' if r2_client else ''}. Xem ở tab 📚 Thư viện.")
        except Exception as e: st.error(str(e))

# ════════════ TAB 2: THƠ & TRUYỆN ════════════
with tab_content:
    st.markdown("### 📝 Tạo Thơ & Câu Chuyện Giáo Dục Mầm Non")
    st.info("💡 AI tạo bài thơ hoặc câu chuyện **mang tính giáo dục** — kèm **kịch bản phân cảnh minh họa** và **hướng dẫn dạy học** chi tiết cho giáo viên.")

    c_cat, c_topic_sel = st.columns([1,2])
    with c_cat:
        category = st.selectbox("📂 Nhóm chủ đề", list(CONTENT_THEMES.keys()), key="ct_category")
    with c_topic_sel:
        topics_in_cat = list(CONTENT_THEMES[category].keys())
        topic_sel = st.selectbox("🎯 Chủ đề cụ thể", topics_in_cat, key="ct_topic")

    skill_goal = CONTENT_THEMES[category][topic_sel]
    st.caption(f"🎯 Mục tiêu giáo dục: *{skill_goal}*")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        ct_age  = st.selectbox("👶 Độ tuổi", AGE_GROUPS, index=1, key="ct_age")
    with c2:
        ct_type = st.radio("📄 Loại nội dung", ["🎭 Bài thơ", "📖 Câu chuyện"], horizontal=True, key="ct_type")

    ct_options = st.multiselect(
        "✅ Tạo thêm (tuỳ chọn)",
        ["🎬 Kịch bản phân cảnh minh họa", "📋 Hướng dẫn dạy học"],
        default=["🎬 Kịch bản phân cảnh minh họa", "📋 Hướng dẫn dạy học"],
        key="ct_options"
    )

    btn_gen_content = st.button("✨ Tạo nội dung", use_container_width=True, key="btn_gen_content")

    if btn_gen_content:
        poem_type = "tho" if "Bài thơ" in ct_type else "truyen"
        try:
            with st.spinner("Đang sáng tác nội dung..."):
                content_ai = generate_poem(topic_sel, ct_age, skill_goal, poem_type)
            st.session_state["ct_content"]    = content_ai
            st.session_state["ct_topic_name"] = topic_sel
            st.session_state["ct_skill_goal"] = skill_goal
            st.session_state["ct_age_sel"]    = ct_age
            st.session_state["ct_type_sel"]   = poem_type
            st.session_state["ct_scenes"]     = ""
            st.session_state["ct_guide"]      = ""

            if "🎬 Kịch bản phân cảnh minh họa" in ct_options:
                with st.spinner("Đang tạo kịch bản phân cảnh..."):
                    st.session_state["ct_scenes"] = generate_scenes(content_ai, topic_sel, poem_type, ct_age)

            if "📋 Hướng dẫn dạy học" in ct_options:
                with st.spinner("Đang tạo hướng dẫn dạy học..."):
                    st.session_state["ct_guide"] = generate_teaching_guide(content_ai, topic_sel, skill_goal, ct_age, poem_type)

            # ── Lưu vào thư viện ngay sau khi tạo ──
            with st.spinner("Đang lưu vào thư viện..."):
                saved_ts, r2_url = save_content_to_library(
                    title=topic_sel,
                    topic=topic_sel,
                    skill_goal=skill_goal,
                    age_group=ct_age,
                    content_type=poem_type,
                    content_text=content_ai,
                    scenes=st.session_state.get("ct_scenes",""),
                    teaching_guide=st.session_state.get("ct_guide",""),
                    category=category,
                )
            st.session_state["ct_saved_ts"] = saved_ts
            if r2_url:
                st.success(f"✅ Đã tạo và lưu vào thư viện! ☁️ Đồng bộ lên R2.")
            else:
                st.success("✅ Đã tạo và lưu vào thư viện!")
        except Exception as e:
            st.error(f"Lỗi: {e}")

    if st.session_state.get("ct_content"):
        st.divider()
        type_label = "📜 Bài thơ" if st.session_state.get("ct_type_sel") == "tho" else "📖 Câu chuyện"
        topic_name = st.session_state.get("ct_topic_name", "")
        content_val = st.session_state.get("ct_content", "")
        scenes_val  = st.session_state.get("ct_scenes", "")
        guide_val   = st.session_state.get("ct_guide", "")

        r_tab1, r_tab2, r_tab3 = st.tabs([f"{type_label}", "🎬 Phân cảnh minh họa", "📋 Hướng dẫn dạy học"])

        with r_tab1:
            st.markdown(f"#### {type_label}: {topic_name}")
            st.caption(f"*Mục tiêu: {st.session_state.get('ct_skill_goal','')} | Độ tuổi: {st.session_state.get('ct_age_sel','')}*")
            st.markdown(content_val)
            st.divider()
            full_text = f"{type_label}: {topic_name}\nMục tiêu: {st.session_state.get('ct_skill_goal','')}\nĐộ tuổi: {st.session_state.get('ct_age_sel','')}\n\n{content_val}"
            if scenes_val: full_text += f"\n\n{'='*50}\nKỊCH BẢN PHÂN CẢNH MINH HỌA\n{'='*50}\n{scenes_val}"
            if guide_val:  full_text += f"\n\n{'='*50}\nHƯỚNG DẪN DẠY HỌC\n{'='*50}\n{guide_val}"
            st.download_button("⬇️ Tải toàn bộ tài liệu (.txt)",
                data=full_text.encode("utf-8"),
                file_name=f"{ascii_slugify(topic_name)}_tron_bo.txt",
                mime="text/plain", use_container_width=True)

        with r_tab2:
            if scenes_val:
                st.markdown("#### 🎬 Kịch bản phân cảnh minh họa")
                st.caption("Dùng để vẽ tranh, in ấn, kể chuyện trực quan hoặc làm flashcard cho trẻ")
                st.markdown(scenes_val)
                st.download_button("⬇️ Tải kịch bản phân cảnh (.txt)",
                    data=scenes_val.encode("utf-8"),
                    file_name=f"{ascii_slugify(topic_name)}_phan_canh.txt",
                    mime="text/plain", use_container_width=True)
            else:
                st.info("Chọn **'🎬 Kịch bản phân cảnh minh họa'** ở trên rồi bấm Tạo nội dung.")

        with r_tab3:
            if guide_val:
                st.markdown("#### 📋 Hướng dẫn sử dụng trong dạy học")
                st.caption("Gợi ý câu hỏi, hoạt động tương tác và trò chơi củng cố kiến thức")
                st.markdown(guide_val)
                st.download_button("⬇️ Tải hướng dẫn dạy học (.txt)",
                    data=guide_val.encode("utf-8"),
                    file_name=f"{ascii_slugify(topic_name)}_huong_dan.txt",
                    mime="text/plain", use_container_width=True)
            else:
                st.info("Chọn **'📋 Hướng dẫn dạy học'** ở trên rồi bấm Tạo nội dung.")

        st.divider()
        st.markdown("#### 🎵 Chuyển thành bài hát?")
        ct_song_style = st.selectbox("Phong cách nhạc", STYLE_DISPLAY, index=0, key="ct_song_style")
        ct_instrumental = st.toggle("Chỉ giai điệu", value=False, key="ct_instrumental")
        if st.button("🎧 Chuyển thành bài hát & Tạo nhạc", use_container_width=True, key="btn_ct_to_song"):
            style_ct = STYLE_MAP.get(ct_song_style.split("–")[0].strip(), DEFAULT_SUNOSTYLE)
            try:
                with st.spinner("Đang chuyển thành lời bài hát..."):
                    converted = poem_to_song(content_val, st.session_state.get("ct_age_sel",""), ct_song_style)
                st.session_state["ct_converted_lyrics"] = converted
                st.session_state["ct_converted_style"]  = style_ct
                st.session_state["ct_song_title"]       = topic_name
                st.success("✅ Đã chuyển thành lời! Xem bên dưới.")
            except Exception as e: st.error(str(e))

        if st.session_state.get("ct_converted_lyrics"):
            st.text_area("Lời bài hát:", value=st.session_state["ct_converted_lyrics"], height=250, key="ct_lyrics_show")
            if st.button("🎶 Tạo nhạc ngay", use_container_width=True, key="btn_ct_music"):
                try:
                    with st.spinner("Đang tạo bài hát..."):
                        task_id = suno_generate_song(
                            st.session_state["ct_converted_lyrics"],
                            st.session_state.get("ct_song_title","Kids Song"),
                            st.session_state.get("ct_converted_style", DEFAULT_SUNOSTYLE),
                            ct_instrumental)
                        tracks = suno_poll(task_id)
                    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                    base = ascii_slugify(st.session_state.get("ct_song_title","Kids_Song"))
                    for i, t in enumerate(tracks, 1):
                        audio_url_orig = t.get("audioUrlHigh") or t.get("audioUrl") or ""; mp3_path = ""; audio_bytes = b""
                        if audio_url_orig:
                            audio_bytes = download_bytes(audio_url_orig)
                            mp3_path = os.path.join(MP3_DIR,f"{ts}_{i}_{base}.mp3"); open(mp3_path,"wb").write(audio_bytes)
                        audio_url_r2 = r2_upload_mp3(f"mp3/{ts}_{i}_{base}.mp3", audio_bytes) if audio_bytes and r2_client else None
                        audio_url_final = audio_url_r2 or audio_url_orig or ""
                        st.write(f"**{st.session_state.get('ct_song_title','Kids Song')} — Bản {i}**")
                        if audio_url_r2: st.caption("☁️ Đã lưu lên Cloudflare R2")
                        if mp3_path and os.path.exists(mp3_path):
                            mp3_data = open(mp3_path,"rb").read(); st.audio(mp3_data, format="audio/mp3")
                            st.download_button("⬇️ Tải MP3", data=mp3_data, file_name=os.path.basename(mp3_path), mime="audio/mpeg", use_container_width=True, key=f"dl_ct_{ts}_{i}")
                        elif audio_url_final: st.audio(audio_url_final, format="audio/mp3")
                        write_history_row({"time":ts,"title":st.session_state.get("ct_song_title","Kids Song"),
                            "topic":topic_name,"keywords":skill_goal,"style":ct_song_style,"language":"vi",
                            "verses":"2","bridge":"true","instrumental":str(ct_instrumental).lower(),
                            "track_index":i,"audio_url":audio_url_final,"image_url":"",
                            "mp3_path":mp3_path if not audio_url_r2 else "","cover_path":"",
                            "lyrics":st.session_state["ct_converted_lyrics"],
                            "age_group":st.session_state.get("ct_age_sel",""),"theme_month":"","source_type":"content"})
                    st.balloons(); st.success("✅ Đã tạo nhạc! Xem ở tab 📚 Thư viện.")
                    st.session_state["ct_converted_lyrics"] = ""
                except Exception as e: st.error(str(e))


# ════════════ TAB 3: THƠ/TRUYỆN → NHẠC ════════════
with tab_poem:
    st.markdown("### 📖 Chuyển Thơ / Câu Chuyện Có Sẵn → Bài Hát")
    st.info("💡 Dán bài thơ, đoạn truyện hoặc nội dung bài học — AI sẽ chuyển thành lời bài hát phù hợp với trẻ mầm non.")
    c_p1, c_p2 = st.columns([2,1])
    with c_p1:
        poem_input = st.text_area("📝 Dán nội dung thơ / truyện:", height=220, placeholder="Ví dụ:\nChú thỏ trắng lông như bông...", key="poem_input")
        poem_title = st.text_input("Tiêu đề bài hát", key="poem_title", placeholder="Chú Thỏ Trắng")
    with c_p2:
        poem_age   = st.selectbox("Độ tuổi", AGE_GROUPS, index=1, key="poem_age")
        poem_style = st.selectbox("Phong cách nhạc", STYLE_DISPLAY, index=0, key="poem_style")

    if st.button("🎵 Chuyển thành bài hát", use_container_width=True, disabled=not bool((poem_input or "").strip()), key="btn_convert_poem"):
        style_for_poem = STYLE_MAP.get(poem_style.split("–")[0].strip(), DEFAULT_SUNOSTYLE)
        try:
            with st.spinner("Đang chuyển đổi..."):
                converted = poem_to_song(poem_input, poem_age, poem_style)
            st.session_state.update({"lyrics":converted,"title":poem_title or "Bài Hát Thiếu Nhi","topic":poem_title or "Chuyển từ thơ/truyện","poem_style_for_api":style_for_poem,"poem_converted":True})
            st.success("✅ Đã chuyển đổi thành công!")
        except Exception as e: st.error(f"Lỗi: {e}")

    if st.session_state.get("poem_converted") and st.session_state.lyrics:
        st.markdown("---")
        st.session_state.lyrics = st.text_area("Chỉnh sửa lời nếu cần:", value=st.session_state.lyrics, height=300, key="poem_lyrics_edit")
        c_r1, c_r2 = st.columns(2)
        with c_r1: poem_instrumental = st.toggle("Chỉ giai điệu", value=False, key="poem_instrumental")
        with c_r2: btn_poem_music = st.button("🎧 Tạo nhạc từ lời này", use_container_width=True, key="btn_poem_music")
        if btn_poem_music:
            try:
                with st.spinner("Đang tạo bài hát..."):
                    task_id = suno_generate_song(st.session_state.lyrics, st.session_state.title or "Kids Song", st.session_state.get("poem_style_for_api",DEFAULT_SUNOSTYLE), poem_instrumental)
                    tracks  = suno_poll(task_id)
                ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S"); base = ascii_slugify(st.session_state.title or "Kids_Song")
                for i, t in enumerate(tracks, 1):
                    audio_url_orig = t.get("audioUrlHigh") or t.get("audioUrl") or ""; mp3_path = ""; audio_bytes = b""
                    if audio_url_orig:
                        audio_bytes = download_bytes(audio_url_orig); mp3_path = os.path.join(MP3_DIR,f"{ts}_{i}_{base}.mp3"); open(mp3_path,"wb").write(audio_bytes)
                    audio_url_r2 = r2_upload_mp3(f"mp3/{ts}_{i}_{base}.mp3", audio_bytes) if audio_bytes and r2_client else None
                    audio_url_final = audio_url_r2 or audio_url_orig or ""
                    st.write(f"**{st.session_state.title} — Bản {i}**")
                    if audio_url_r2: st.caption("☁️ Đã lưu lên Cloudflare R2")
                    if mp3_path and os.path.exists(mp3_path):
                        mp3_data = open(mp3_path,"rb").read(); st.audio(mp3_data, format="audio/mp3")
                        st.download_button("⬇️ Tải MP3", data=mp3_data, file_name=os.path.basename(mp3_path), mime="audio/mpeg", use_container_width=True, key=f"dl_poem_{ts}_{i}")
                    elif audio_url_final: st.audio(audio_url_final, format="audio/mp3")
                    write_history_row({"time":ts,"title":st.session_state.title or "Kids Song","topic":poem_title or "Thơ/truyện","keywords":"","style":poem_style,"language":"vi","verses":"","bridge":"true","instrumental":str(poem_instrumental).lower(),"track_index":i,"audio_url":audio_url_final,"image_url":t.get("imageUrl") or "","mp3_path":mp3_path if not audio_url_r2 else "","cover_path":"","lyrics":st.session_state.lyrics,"age_group":poem_age,"theme_month":"","source_type":"poem"})
                st.balloons(); st.success("✅ Đã lưu bài hát! Xem ở tab 📚 Thư viện."); st.session_state["poem_converted"] = False
            except Exception as e: st.error(str(e))

# ════════════ TAB 4: THƯ VIỆN ════════════
with tab_library:
    st.session_state.setdefault("lib_page", 1)
    st.markdown("### 📚 Thư viện — Tất cả sản phẩm đã tạo")

    df = load_history_df_local()
    # Đảm bảo các cột mới tồn tại
    for col in ["content_type","content_text","scenes","teaching_guide","skill_goal"]:
        if col not in df.columns: df[col] = ""

    if df is None or len(df) == 0:
        st.info("Chưa có sản phẩm nào trong thư viện.")
    else:
        # ── Bộ lọc ──
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        with c1: q = st.text_input("🔍 Tìm kiếm tiêu đề / chủ đề", key="lib_kw").strip()
        with c2:
            loai_filter = st.selectbox("📂 Loại sản phẩm",
                ["Tất cả", "🎵 Bài hát", "📜 Bài thơ", "📖 Câu chuyện"], key="lib_loai")
        with c3:
            av = sorted(df["age_group"].dropna().unique().tolist()) if "age_group" in df.columns else []
            ap = st.selectbox("👶 Độ tuổi", ["Tất cả"]+av, key="lib_age")
        with c4:
            so = st.selectbox("📅 Sắp xếp", ["Mới nhất","A→Z"], key="lib_sort")

        # Lọc theo loại
        if loai_filter == "🎵 Bài hát":
            df = df[~df["source_type"].isin(["content"])]
        elif loai_filter == "📜 Bài thơ":
            df = df[(df["source_type"]=="content") & (df["content_type"]=="tho")]
        elif loai_filter == "📖 Câu chuyện":
            df = df[(df["source_type"]=="content") & (df["content_type"]=="truyen")]

        if q:
            qn = norm_txt(q); mask = pd.Series(False, index=df.index)
            for col in ["title","topic","content_text"]:
                if col in df.columns: mask |= df[col].astype(str).map(norm_txt).str.contains(qn, na=False)
            df = df[mask]
        if ap != "Tất cả" and "age_group" in df.columns:
            df = df[df["age_group"]==ap]
        if "time" in df.columns: df["time_dt"] = df["time"].apply(parse_time_safe)
        if so == "Mới nhất" and "time_dt" in df.columns:
            df = df.sort_values("time_dt", ascending=False, na_position="last")
        elif so == "A→Z" and "title" in df.columns:
            df = df.sort_values("title", key=lambda s: s.astype(str).str.lower())

        total = len(df)
        # Đếm theo loại để hiện badge
        n_song    = len(df[~df["source_type"].isin(["content"])]) if "source_type" in df.columns else 0
        n_tho     = len(df[(df.get("source_type","")=="content") & (df.get("content_type","")=="tho")]) if "content_type" in df.columns else 0
        n_truyen  = len(df[(df.get("source_type","")=="content") & (df.get("content_type","")=="truyen")]) if "content_type" in df.columns else 0
        st.markdown(
            f"**Tổng: {total} sản phẩm** &nbsp;|&nbsp; "
            f"🎵 {n_song} bài hát &nbsp;|&nbsp; 📜 {n_tho} bài thơ &nbsp;|&nbsp; 📖 {n_truyen} câu chuyện",
            unsafe_allow_html=True)

        if total == 0:
            st.info("Không có sản phẩm nào khớp bộ lọc.")
        else:
            page_size = st.selectbox("Số sản phẩm/trang", [8,12,16,24], key="lib_pagesize")
            page_count = max(1,(total+page_size-1)//page_size)
            st.session_state.lib_page = st.number_input("Trang", 1, page_count,
                min(st.session_state.lib_page, page_count), key="lib_page_input")
            start = (st.session_state.lib_page-1)*page_size
            df_page = df.iloc[start:start+page_size].reset_index(drop=True)

            for idx, row in df_page.iterrows():
                src = str(row.get("source_type",""))
                ctype = str(row.get("content_type",""))
                is_content = (src == "content")

                if is_content:
                    # ── Hiển thị Thơ / Truyện ──
                    icon = "📜" if ctype == "tho" else "📖"
                    label = "Bài thơ" if ctype == "tho" else "Câu chuyện"
                    with st.expander(f"{icon} **{label}:** {row.get('title','')} &nbsp;·&nbsp; {str(row.get('time',''))[:15]}", expanded=False):
                        col_info, col_dl = st.columns([2,1])
                        with col_info:
                            st.caption(f"🎯 Mục tiêu: {row.get('skill_goal', row.get('keywords',''))}")
                            st.caption(f"👶 Độ tuổi: {row.get('age_group','')}  |  📂 Nhóm: {row.get('style','')}")
                        with col_dl:
                            # Tải toàn bộ tài liệu
                            full = str(row.get("content_text",""))
                            sep = "=" * 40
                            if row.get("scenes"): full += "\n\n" + sep + "\nPHAN CANH\n" + sep + "\n" + str(row["scenes"])
                            if row.get("teaching_guide"): full += "\n\n" + sep + "\nHUONG DAN DAY HOC\n" + sep + "\n" + str(row["teaching_guide"])
                            if full.strip():
                                st.download_button("⬇️ Tải tài liệu (.txt)",
                                    data=full.encode("utf-8"),
                                    file_name=f"{ascii_slugify(row.get('title','noi_dung'))}.txt",
                                    mime="text/plain",
                                    key=f"dl_content_{idx}_{row.get('time','')}")
                        # Hiển thị nội dung
                        tab_a, tab_b, tab_c = st.tabs([f"{icon} Nội dung", "🎬 Phân cảnh", "📋 Hướng dẫn"])
                        with tab_a:
                            st.markdown(str(row.get("content_text","")) or "_Chưa có nội dung_")
                        with tab_b:
                            st.markdown(str(row.get("scenes","")) or "_Chưa có kịch bản phân cảnh_")
                        with tab_c:
                            st.markdown(str(row.get("teaching_guide","")) or "_Chưa có hướng dẫn dạy học_")
                else:
                    # ── Hiển thị Bài hát ──
                    with st.expander(f"🎵 **Bài hát:** {row.get('title','Kids Song')} &nbsp;·&nbsp; {str(row.get('time',''))[:15]}", expanded=False):
                        c_img, c_meta = st.columns([1, 2.5])
                        with c_img: show_cover_from_row(row)
                        with c_meta:
                            st.write(f"**Chủ đề:** {row.get('topic','')}")
                            st.write(f"**Phong cách:** {row.get('style','')}")
                            st.write(f"**Độ tuổi:** {row.get('age_group','')}")
                            show_audio_from_row(row, key_suffix=f"lib_{idx}")
                st.divider()

            if st.button("🔄 Làm mới thư viện", use_container_width=True): st.rerun()

# ════════════ TAB 5: LỊCH SỬ ════════════
with tab_history:
    st.subheader("🗂️ Lịch sử các bài hát đã tạo")
    try: df_all = load_history_df_local()
    except Exception as e: st.warning(f"Lỗi: {e}"); df_all = pd.DataFrame(columns=EXPECTED_HEADER)
    if df_all is None or df_all.empty: st.info("Chưa có dữ liệu lịch sử.")
    else:
        rename = {"time":"Thời gian","title":"Tên bài hát","topic":"Chủ đề / Gợi ý","style":"Phong cách nhạc","audio_url":"Liên kết nhạc","track_index":"Bản số","language":"Ngôn ngữ"}
        for k in rename:
            if k not in df_all.columns: df_all[k] = ""
        df_all = df_all.rename(columns=rename)
        df_all["Thời gian_dt"] = df_all["Thời gian"].apply(parse_time_safe)
        df_all = df_all.sort_values("Thời gian_dt", ascending=False).reset_index(drop=True)
        c1, c2 = st.columns(2)
        with c1:
            vt = df_all["Thời gian_dt"].dropna()
            date_range = st.date_input("Khoảng thời gian",(vt.min().date(),vt.max().date())) if len(vt)>0 else None
        with c2:
            so = ["Tất cả"]+sorted([s for s in df_all["Phong cách nhạc"].dropna().astype(str).unique() if s.strip()])
            sf = st.selectbox("Phong cách nhạc", so)
        dff = df_all.copy()
        if isinstance(date_range,(list,tuple)) and len(date_range)==2:
            s,e = date_range; ht = dff["Thời gian_dt"].notna()
            ir = (dff["Thời gian_dt"]>=dt.datetime.combine(s,dt.time.min))&(dff["Thời gian_dt"]<=dt.datetime.combine(e,dt.time.max))
            dff = pd.concat([dff[ht&ir],dff[~ht]],ignore_index=True)
        if sf != "Tất cả": dff = dff[dff["Phong cách nhạc"].astype(str)==sf]
        st.success(f"Hiển thị {len(dff)} bản ghi.")
        cs = [c for c in ["Thời gian","Tên bài hát","Chủ đề / Gợi ý","Phong cách nhạc","Ngôn ngữ","Bản số","Liên kết nhạc"] if c in dff.columns]
        st.dataframe(dff[cs], use_container_width=True, height=420)
        st.download_button("Tải về CSV", data=dff.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"lich_su_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)

# ════════════ TAB 6: CÀI ĐẶT ════════════
with tab_settings:
    if r2_client:
        st.success(f"✅ Đã kết nối Cloudflare R2 — Bucket: `{R2_BUCKET_NAME}`")
        c1, c2 = st.columns(2)
        with c1: btn_list_r2 = st.button("🔎 Xem file MP3 trong R2")
        with c2: btn_r2_info = st.button("ℹ️ Thông tin bucket")
        if btn_list_r2:
            files = r2_list_files("mp3/"); mp3s = [f for f in files if str(f.get("Key","")).endswith(".mp3")]
            st.write(f"📁 Có **{len(mp3s)} file MP3** trong bucket `{R2_BUCKET_NAME}`")
            for fi in mp3s[:20]:
                key = fi.get("Key",""); url = r2_get_public_url(key)
                st.markdown(f"- 🎵 `{key}` ({fi.get('Size',0)//1024} KB)"+( f" → [nghe]({url})" if url else ""))
            if not mp3s: st.info("Bucket chưa có file MP3 nào.")
        if btn_r2_info:
            st.info(f"**Cloudflare R2:**\n- Bucket: `{R2_BUCKET_NAME}`\n- Account: `{(R2_ACCOUNT_ID or '')[:8]}...`\n- Public URL: `{R2_PUBLIC_URL or 'Chưa cấu hình'}`")
    else:
        st.warning("⚠️ Chưa kết nối Cloudflare R2.")
        st.code("R2_ACCOUNT_ID = \"your_account_id\"\nR2_ACCESS_KEY_ID = \"your_key_id\"\nR2_SECRET_ACCESS_KEY = \"your_secret\"\nR2_BUCKET_NAME = \"kids-songs\"\nR2_PUBLIC_URL = \"https://pub-xxxxx.r2.dev\"")
    st.divider()
    st.markdown("### ℹ️ Ghi chú")
    st.markdown(
        "- Tab **📝 Thơ & Truyện**: AI tạo thơ/truyện kèm hình minh họa, có thể chuyển thành nhạc.\n"
        "- **DALL-E** vẽ hình tốn ~$0.04/ảnh, ảnh có sẵn miễn phí.\n"
        "- **Refine** chỉnh câu từ, không đổi chủ đề.\n"
        "- Thư viện và lịch sử đọc từ `outputs/tracks.csv`.")

st.markdown("""<hr style="margin:24px 0;border:none;border-top:1px solid #e6e8f5;">
<div style="text-align:center;line-height:1.7;">
  <div style="font-weight:800;font-size:18px;">© MẦM NON STUDIO • Dành cho Giáo viên mầm non</div>
  <div style="font-size:15px;color:#64748b;">Facebook: Ngọc Thảo – <a href="mailto:ms.nthaotran@gmail.com">ms.nthaotran@gmail.com</a></div>
</div>""", unsafe_allow_html=True)
