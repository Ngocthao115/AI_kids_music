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

# ── Cloudflare R2 (thay thế Supabase Storage) ──
R2_ACCOUNT_ID        = get_secret("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID     = get_secret("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = get_secret("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME       = get_secret("R2_BUCKET_NAME", "kids-songs")
R2_PUBLIC_URL        = get_secret("R2_PUBLIC_URL", "")   # VD: https://pub-xxxxx.r2.dev

ANALYTICS_CSV = get_secret("ANALYTICS_CSV", "du_lieu_tao_nhac.csv")

if not OPENAI_API_KEY: st.error("Thiếu OPENAI_API_KEY"); st.stop()
if not SUNO_API_KEY:   st.error("Thiếu SUNO_API_KEY"); st.stop()
if not SUNO_CALLBACK_URL: st.warning("Chưa có SUNO_CALLBACK_URL — app vẫn hoạt động bằng cách poll kết quả.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client  = OpenAI(api_key=OPENAI_API_KEY)
HEADERS = {"Authorization": f"Bearer {SUNO_API_KEY}", "Content-Type": "application/json"}

# ── Khởi tạo R2 client ──
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
        msg = f"Bucket '{R2_BUCKET_NAME}' chưa tồn tại." if code == "404" else f"Lỗi R2: {e}"
        st.warning(msg); r2_client = None
    except Exception as e:
        st.warning(f"Không kết nối được R2: {e}"); r2_client = None

OUTPUT_DIR = "outputs"; MP3_DIR = os.path.join(OUTPUT_DIR,"mp3"); COVER_DIR = os.path.join(OUTPUT_DIR,"covers")
os.makedirs(MP3_DIR, exist_ok=True); os.makedirs(COVER_DIR, exist_ok=True)
HISTORY_CSV = os.path.join(OUTPUT_DIR, "tracks.csv")
EXPECTED_HEADER = ["time","title","topic","keywords","style","language","verses","bridge",
    "instrumental","track_index","audio_url","image_url","mp3_path","cover_path",
    "lyrics","age_group","theme_month","source_type"]

DEFAULT_LYRICS_SYSTEM = (
    "Bạn là một chuyên gia sáng tác nhạc thiếu nhi và là một nhà sư phạm mầm non giàu kinh nghiệm, "
    "bạn hiểu rõ tâm lý trẻ em từ 3-6 tuổi và có khả năng biến các bài học giáo dục thành lời ca trong sáng, dễ thuộc, dễ nhớ. "
    "Hãy sáng tác lời cho một bài hát thiếu nhi. Mỗi câu hát chỉ được có từ 5 đến 10 từ. "
    "Ngôn ngữ mang tính giáo dục, ý nghĩa nhân văn. Vần điệu rõ ràng, có điệp khúc dễ nhớ. "
    "YÊU CẦU: Mặc định dùng nhãn [Verse] và [Chorus]. CHỈ dùng cấu trúc rap khi yêu cầu rõ.")

# ── Hàm nghiệp vụ ──
def build_user_prompt(topic, language="vi", target_words=None, verses=2, include_bridge=True, min_lines=12, max_lines=18):
    tw = ", ".join(target_words) if target_words else "Không bắt buộc"
    structure = ["- Cấu trúc: [Verse 1] → [Chorus]"] + [f"→ [Verse {i}] → [Chorus]" for i in range(2, verses+1)]
    if include_bridge: structure.append("→ [Bridge] (2–4 dòng) → [Chorus] (kết)")
    return f"Chủ đề: {topic}\nNgôn ngữ: {language}\n{' '.join(structure)}.\nTừ khóa: {tw}\nĐộ dài ~{min_lines}–{max_lines} dòng.\n"

def generate_lyrics(topic, target_words=None, language="vi", verses=2, bridge=True):
    r = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"system","content":DEFAULT_LYRICS_SYSTEM},
                  {"role":"user","content":build_user_prompt(topic,language,target_words,verses,bridge).strip()}],
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
        messages=[{"role":"system","content":"Bạn là chuyên gia chuyển thơ thiếu nhi thành lời bài hát. Cấu trúc [Verse]/[Chorus]/[Bridge], vần điệu rõ, câu ngắn 5-10 từ."},
                  {"role":"user","content":f"Lứa tuổi: {age_group}\nPhong cách: {style_hint or 'Tươi vui'}\n\nNội dung gốc:\n{poem_text}\n\nChuyển thành lời bài hát thiếu nhi."}],
        temperature=0.8, max_tokens=800)
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
    if data.get("code") != 200 or not data.get("data",{}).get("taskId"): raise RuntimeError("Suno failed: " + json.dumps(data))
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

# ── Hàm Cloudflare R2 (thay thế Supabase Storage) ──
def r2_upload_bytes(object_key, data_bytes, content_type):
    """Upload file lên R2. Trả về public URL hoặc presigned URL."""
    if not r2_client or not data_bytes: return None
    try:
        r2_client.put_object(Bucket=R2_BUCKET_NAME, Key=object_key, Body=data_bytes, ContentType=content_type)
        if R2_PUBLIC_URL: return f"{R2_PUBLIC_URL.rstrip('/')}/{object_key}"
        return r2_client.generate_presigned_url("get_object",
            Params={"Bucket":R2_BUCKET_NAME,"Key":object_key}, ExpiresIn=604800)
    except Exception as e:
        st.warning(f"Upload R2 thất bại ({object_key}): {e}"); return None

def r2_upload_mp3(object_key, mp3_bytes): return r2_upload_bytes(object_key, mp3_bytes, "audio/mpeg")

def r2_list_files(prefix="mp3/"):
    if not r2_client: return []
    try: return r2_client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix).get("Contents", [])
    except Exception as e: st.warning(f"Lỗi liệt kê R2: {e}"); return []

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
    cols = ["Thời gian","Tên bài hát","Miêu tả bài hát","Từ ngữ gợi ý","Chủ đề / Gợi ý",
            "Phong cách nhạc","Ngôn ngữ","Số verse","Bridge","Liên kết nhạc","Liên kết lời hát"]
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
    else: st.image("https://picsum.photos/seed/kidsmusic/600/400", use_container_width=True)

def show_audio_from_row(row, key_suffix=""):
    mp = str(row.get("mp3_path","") or "").strip(); au = str(row.get("audio_url","") or "").strip()
    if mp and os.path.exists(mp):
        data = open(mp,"rb").read()
        st.audio(data, format="audio/mp3")
        st.download_button("⬇ Tải MP3", data=data, file_name=os.path.basename(mp), mime="audio/mpeg",
            use_container_width=True, key=f"dl_{row.get('time','')}_{row.get('track_index','1')}_{key_suffix}")
    elif au: st.audio(au, format="audio/mp3")

# ── UI ──
st.set_page_config(page_title="NHẠC AI THIẾU NHI - MẦM NON", page_icon="🎵", layout="centered")

def add_bg_from_local(image_path, alpha=0.85, size="cover", position="center"):
    try:
        encoded = base64.b64encode(open(image_path,"rb").read()).decode()
        st.markdown(f"<style>[data-testid='stAppViewContainer']{{background-image:linear-gradient(rgba(255,255,255,{alpha}),rgba(255,255,255,{alpha})),url('data:image/png;base64,{encoded}');background-size:{size};background-position:{position};background-repeat:no-repeat;}}</style>",unsafe_allow_html=True)
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
</style>""", unsafe_allow_html=True)

for k,v in [("lyrics",""),("title",""),("topic",""),("targets",[]),("generated",False),("keywords",""),("language","vi"),("verses",2),("bridge",True)]:
    st.session_state.setdefault(k, v)

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

with st.sidebar:
    st.markdown("## 👩‍🏫 Hướng dẫn nhanh")
    st.markdown("**✨ Tạo bài hát:**\n1. Chọn chủ đề → Tự động điền\n2. Chọn độ tuổi, phong cách\n3. Tạo lời → Tạo nhạc → Tải MP3\n\n**📖 Từ thơ/truyện:**\n1. Tab Thơ/Truyện → Nhạc\n2. Dán nội dung → Chuyển → Tạo nhạc")
    st.divider()
    st.caption(f"Model Suno: **{SUNO_MODEL}**")
    st.caption(f"Cloudflare R2: **{r2_status}**")

st.title("🎵 NHẠC AI THIẾU NHI - MẦM NON")
st.markdown('<span class="badge">🏫 Dành riêng cho Giáo viên Mầm non</span>&nbsp;<span class="badge">✨ Tạo nhạc thiếu nhi bằng AI</span>&nbsp;<span class="badge">☁️ Lưu trữ Cloudflare R2</span>', unsafe_allow_html=True)

tab_make, tab_poem, tab_library, tab_stats, tab_history, tab_settings = st.tabs(["✨ Tạo bài hát","📖 Thơ/Truyện → Nhạc","📚 Thư viện","📊 Thống kê","🗂️ Lịch sử","⚙️ Cài đặt"])

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
    with c2: refine_hint = st.text_input("Chỉ dẫn refine", placeholder="Ví dụ: nhịp nhanh hơn…")
    with c3: btn_refine  = st.button("🪄 Refine", use_container_width=True, disabled=not bool(st.session_state.lyrics.strip()))

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

with tab_poem:
    st.markdown("### 📖 Chuyển Thơ / Câu Chuyện → Bài Hát Thiếu Nhi")
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
                    audio_url_r2   = r2_upload_mp3(f"mp3/{ts}_{i}_{base}.mp3", audio_bytes) if audio_bytes and r2_client else None
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

with tab_library:
    st.session_state.setdefault("lib_page", 1)
    st.markdown("### 📚 Thư viện (Gallery)")
    df = load_history_df_local()
    if df is None or len(df) == 0:
        st.info("Chưa có bài nhạc nào trong thư viện.")
    else:
        c1,c2,c3,c4 = st.columns([1.2,1,1,1])
        with c1: q = st.text_input("Tìm kiếm", key="lib_kw").strip()
        with c2:
            sv = sorted(df["style"].dropna().unique().tolist()) if "style" in df.columns else []
            sp = st.selectbox("Phong cách", ["Tất cả"]+sv, key="lib_style")
        with c3:
            av = sorted(df["age_group"].dropna().unique().tolist()) if "age_group" in df.columns else []
            ap = st.selectbox("Độ tuổi", ["Tất cả"]+av, key="lib_age")
        with c4: so = st.selectbox("Sắp xếp", ["Mới nhất","A→Z","Theo style"], key="lib_sort")
        if q:
            qn = norm_txt(q); mask = pd.Series(False, index=df.index)
            for col in ["title","topic"]:
                if col in df.columns: mask |= df[col].astype(str).map(norm_txt).str.contains(qn, na=False)
            df = df[mask]
        if sp != "Tất cả" and "style" in df.columns:    df = df[df["style"]==sp]
        if ap != "Tất cả" and "age_group" in df.columns: df = df[df["age_group"]==ap]
        if "time" in df.columns: df["time_dt"] = df["time"].apply(parse_time_safe)
        if so == "Mới nhất" and "time_dt" in df.columns: df = df.sort_values("time_dt", ascending=False, na_position="last")
        elif so == "A→Z" and "title" in df.columns: df = df.sort_values("title", key=lambda s: s.astype(str).str.lower())
        if len(df) == 0: st.info("Không có bài nào khớp bộ lọc.")
        else:
            cv1, cv2 = st.columns(2)
            with cv1: layout_mode = st.radio("Kiểu hiển thị", ["Danh sách","Lưới"], horizontal=True)
            with cv2: page_size = st.selectbox("Số bài/trang", [8,12,16,24])
            total = len(df); page_count = max(1,(total+page_size-1)//page_size)
            st.caption(f"Tổng {total} bài")
            st.session_state.lib_page = st.number_input("Trang", 1, page_count, min(st.session_state.lib_page, page_count), key="lib_page_input")
            start = (st.session_state.lib_page-1)*page_size; df_page = df.iloc[start:start+page_size].reset_index(drop=True)
            if layout_mode == "Lưới":
                cols = st.columns(4)
                for idx, row in df_page.iterrows():
                    with cols[idx%4]:
                        show_cover_from_row(row); st.markdown(f"**{row.get('title') or 'Kids Song'}**"); st.caption(str(row.get("time",""))); show_audio_from_row(row, key_suffix=f"grid_{idx}")
            else:
                for idx, row in df_page.iterrows():
                    ci, cm = st.columns([1,2.4])
                    with ci: show_cover_from_row(row)
                    with cm:
                        st.markdown(f"**{row.get('title') or 'Kids Song'}**"); st.caption(str(row.get("time",""))); st.write(f"**Chủ đề:** {row.get('topic','')}"); st.write(f"**Phong cách:** {row.get('style','')}"); show_audio_from_row(row, key_suffix=f"list_{idx}")
                    st.divider()
            if st.button("🔄 Làm mới thư viện", use_container_width=True): st.rerun()

with tab_stats:
    st.markdown("### 📊 Thống kê sử dụng")
    dfs = load_history_df_local()
    if dfs is None or dfs.empty: st.info("Chưa có dữ liệu thống kê.")
    else:
        total = len(dfs); tong_tho = len(dfs[dfs["source_type"]=="poem"]) if "source_type" in dfs.columns else 0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🎵 Tổng bài hát", total); c2.metric("✨ Tạo mới", total-tong_tho); c3.metric("📖 Từ thơ/truyện", tong_tho)
        c4.metric("🎼 Phong cách", dfs["style"].nunique() if "style" in dfs.columns else 0)
        st.divider()
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown("#### 🏆 Chủ đề phổ biến")
            if "topic" in dfs.columns:
                for i,(t,c) in enumerate(dfs["topic"].value_counts().head(8).items(),1): st.write(f"{i}. **{t}** — {c} bài")
        with cs2:
            st.markdown("#### 🎼 Phong cách phổ biến")
            if "style" in dfs.columns:
                for sn,c in dfs["style"].value_counts().head(6).items(): st.write(f"**{str(sn)[:30]}** {'█'*min(c*2,20)} {c}")

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
        st.download_button("Tải về CSV", data=dff.to_csv(index=False).encode("utf-8-sig"), file_name=f"lich_su_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)

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
            st.info(f"**Cloudflare R2:**\n- Bucket: `{R2_BUCKET_NAME}`\n- Account: `{(R2_ACCOUNT_ID or '')[:8]}...`\n- Public URL: `{R2_PUBLIC_URL or 'Chưa cấu hình'}`\n- Endpoint: `https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`")
    else:
        st.warning("⚠️ Chưa kết nối Cloudflare R2. Thêm các biến sau vào Secrets:")
        st.code("R2_ACCOUNT_ID = \"your_account_id\"\nR2_ACCESS_KEY_ID = \"your_key_id\"\nR2_SECRET_ACCESS_KEY = \"your_secret\"\nR2_BUCKET_NAME = \"kids-songs\"\nR2_PUBLIC_URL = \"https://pub-xxxxx.r2.dev\"  # tuỳ chọn")
    st.divider()
    st.markdown("### 🎨 Preset chủ đề nhanh")
    preset = st.selectbox("Chọn nhanh",["Màu sắc cơ bản","Hình tròn – vuông – tam giác","Số đếm 1 – 10","Vệ sinh răng miệng","Chào hỏi & phép lịch sự","An toàn giao thông","Con vật","Gia đình","Nghề nghiệp","Trường mầm non","Bản thân bé","Thầy cô và bạn bè"])
    st.caption(f"Chủ đề gợi ý: {preset}")
    st.divider()
    st.markdown("### ℹ️ Ghi chú")
    st.markdown("- **Refine** chỉnh câu từ, không đổi chủ đề.\n- **Instrumental** chỉ tạo giai điệu không lời.\n- Thư viện và lịch sử đọc từ `outputs/tracks.csv`.")

st.markdown("""<hr style="margin:24px 0;border:none;border-top:1px solid #e6e8f5;">
<div style="text-align:center;line-height:1.7;">
  <div style="font-weight:800;font-size:18px;">© NHẠC AI THIẾU NHI • Dành cho Giáo viên mầm non</div>
  <div style="font-size:15px;color:#64748b;">Facebook: Ngọc Thảo – <a href="mailto:ms.nthaotran@gmail.com">ms.nthaotran@gmail.com</a></div>
</div>""", unsafe_allow_html=True)
