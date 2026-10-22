import base64
import html
import inspect
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from rag import recommend_with_candidates

try:
    with open("movies.json", encoding="utf-8") as f:
        MOVIE_COUNT = f"{len(json.load(f)):,}"
except Exception:
    MOVIE_COUNT = "-"

# ---------------------------------------------------------------------------
# Design System (สไตล์เรียบแบบ ChatGPT)
#   พื้นหลัง #FFFFFF / #F7F7F8 · ข้อความ #202123 / #6B7280 · สีเน้น #10A37F
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue="neutral",
    neutral_hue="neutral",
    font=[
        "Inter",
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "Sukhumvit Set",
        "Leelawadee UI",
        "Noto Sans Thai",
        "sans-serif",
    ],
)

CSS = """
:root {
    color-scheme: light;
    --bg: #FFFFFF;
    --bg-subtle: #F7F7F8;
    --text: #202123;
    --text-muted: #6B7280;
    --border: #E5E7EB;
    --border-strong: #D1D5DB;
    --accent: #10A37F;
    --accent-soft: rgba(16, 163, 127, 0.14);
    --danger: #B42318;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-pill: 24px;
    --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px;
    --ease: cubic-bezier(0.4, 0, 0.2, 1);
    --dur: 160ms;
}

/* ---------- Base ---------- */
html, body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, 'Sukhumvit Set', 'Leelawadee UI', 'Noto Sans Thai', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    margin: 0 !important;
    padding: 0 !important;
    animation: none !important;
}
.gradio-container {
    max-width: 100% !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrapper,
.gradio-container .bubble-wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container p, .gradio-container label, .gradio-container .prose { color: var(--text) !important; }
footer { display: none !important; }

/* บังคับตัวแปรสีของ Gradio ให้เป็นโทนสว่างเสมอ (กันกล่องเข้มโผล่เมื่อเป็น dark mode) */
:root, .dark, .gradio-container, .gradio-container.dark {
    --body-background-fill: #FFFFFF;
    --background-fill-primary: #FFFFFF;
    --background-fill-secondary: #F7F7F8;
    --block-background-fill: transparent;
    --block-border-color: transparent;
    --panel-background-fill: transparent;
    --input-background-fill: #FFFFFF;
    --input-background-fill-focus: #FFFFFF;
    --input-border-color: #D1D5DB;
    --input-shadow: none;
    --body-text-color: #202123;
    --body-text-color-subdued: #6B7280;
    --border-color-primary: #E5E7EB;
}

/* ตัวห่อช่องพิมพ์ต้องโปร่งใส ไม่มีพื้นหลังหรือเงา */
.gradio-container form,
.gradio-container .full-container,
.gradio-container .input-container,
.gradio-container .multimodal-textbox,
.gradio-container [data-testid="textbox"],
.gradio-container .block:has(textarea),
.gradio-container .container:has(textarea) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px !important; }

/* Gradio 6 ใส่ padding 32px รอบแอป ทำให้ Sidebar ไม่ชิดขอบและเนื้อหาล้นบนมือถือ */
.gradio-container .main.app, .gradio-container main.fillable { padding: 0 !important; max-width: 100% !important; }

/* ---------- Layout ---------- */
#app-row { gap: 0 !important; align-items: stretch !important; flex-wrap: nowrap !important; }

#sidebar {
    background: var(--bg-subtle) !important;
    border-right: 1px solid var(--border) !important;
    min-height: 100vh;
    padding: var(--s5) var(--s4) !important;
    position: sticky; top: 0; align-self: flex-start;
}
#sidebar-inner { display: flex; flex-direction: column; gap: var(--s5); }

#main { padding: 0 var(--s4) !important; }
#chat-wrapper { max-width: 768px; margin: 0 auto; width: 100%; padding: var(--s5) 0 var(--s4); }

/* ---------- Sidebar ---------- */
.brand { display: flex; align-items: center; gap: var(--s2); font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
.brand svg { width: 20px; height: 20px; flex: none; color: var(--accent); }

.new-chat {
    display: flex; align-items: center; gap: var(--s2);
    padding: 10px var(--s3);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    background: var(--bg);
    color: var(--text) !important;
    font-size: 14px; font-weight: 500;
    text-decoration: none !important;
    transition: background var(--dur) var(--ease), border-color var(--dur) var(--ease);
}
.new-chat svg { width: 16px; height: 16px; flex: none; }
.new-chat:hover { background: #F0F0F2; border-color: #B8BDC6; }

/* ---------- Header ---------- */
.page-title { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; text-align: center; margin: 0 0 var(--s1); color: var(--text); }
.page-subtitle { font-size: 14px; text-align: center; color: var(--text-muted); margin: 0 0 var(--s5); }

/* ---------- Messages ---------- */
.gradio-container .message, .gradio-container .message-content { font-size: 15px !important; line-height: 1.7 !important; }

.message.user, [data-testid="user"] {
    background: var(--bg-subtle) !important; color: var(--text) !important;
    border: none !important; border-radius: 18px !important;
    box-shadow: none !important; backdrop-filter: none !important;
    padding: 10px var(--s4) !important;
}
.message.bot, [data-testid="bot"] {
    background: transparent !important; color: var(--text) !important;
    border: none !important; box-shadow: none !important; backdrop-filter: none !important;
    padding: var(--s2) 0 !important;
}
.message.user *, [data-testid="user"] *,
.message.bot *, [data-testid="bot"] * { color: var(--text) !important; }
.message hr { border: none; border-top: 1px solid var(--border); margin: var(--s4) 0; }

.empty-state { text-align: center; padding: var(--s6) var(--s4); }
.empty-title { font-size: 18px; font-weight: 600; color: var(--text); margin-bottom: var(--s2); }
.empty-text { font-size: 14px; color: var(--text-muted); line-height: 1.6; }

.error-box {
    padding: var(--s3) var(--s4); border: 1px solid #F3C6C1; background: #FEF3F2;
    border-radius: var(--radius-sm); font-size: 14px; line-height: 1.6;
}
.message .error-box, .message .error-box * { color: var(--danger) !important; }

/* ข้อความคำตอบใช้ความกว้างเต็มคอลัมน์ ไม่ถูกบีบตามความกว้างของเนื้อหา */
.gradio-container .message-row.bot-row { width: calc(100% - 20px) !important; align-self: stretch !important; max-width: 100% !important; }
.gradio-container .message-row.bot-row .message,
.gradio-container .message-row.bot-row .flex-wrap,
.gradio-container .message-row.bot-row .role { width: 100% !important; max-width: 100% !important; }
.gradio-container .message-row.bot-row, .gradio-container .message-row.bot-row * { box-sizing: border-box; }
.gradio-container .message-row.bot-row .message-content,
.gradio-container .message-row.bot-row .prose,
.gradio-container .message-row.bot-row .prose * { max-width: 100% !important; overflow-wrap: anywhere; word-break: break-word; white-space: normal; }
.gradio-container .message-row.bot-row .prose.chatbot { display: block; }
/* แถบไอคอนลอยด้านบนของแชต (แชร์/ลบ/คัดลอกทั้งหมด) ทับข้อความ ซ่อนไว้ ใช้ปุ่ม "แชทใหม่" แทน */
.gradio-container .icon-button-wrapper.top-panel { display: none !important; }

/* ---------- Posters ---------- */
/* ล็อกขนาดการ์ดโปสเตอร์ ไม่ให้สไตล์รูปภาพของ Gradio (กว้างเต็มช่อง/ขยายเมื่อ hover) มาทับ */
.gradio-container .message .poster-grid {
    display: flex !important; flex-wrap: wrap; gap: var(--s4); align-items: flex-start;
    margin: var(--s3) 0; padding: 0;
}
.gradio-container .message .poster-item {
    flex: 0 0 108px; width: 108px !important; max-width: 108px; margin: 0; padding: 0;
    position: relative; transition: transform var(--dur) var(--ease);
}
.gradio-container .message .poster-item:hover { transform: translateY(-2px); }
.gradio-container .message img.poster-img {
    display: block; width: 108px !important; min-width: 108px !important; max-width: 108px !important;
    height: 160px !important; min-height: 160px !important; max-height: 160px !important;
    object-fit: cover; margin: 0 !important; padding: 0 !important;
    border-radius: var(--radius-sm); border: 1px solid var(--border); background: var(--bg-subtle);
    transform: none !important; cursor: default; opacity: 1 !important;
}
.poster-title {
    margin-top: 6px; font-size: 12px; font-weight: 500; line-height: 1.4;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.poster-genre {
    margin-top: 2px; font-size: 11px; line-height: 1.4; color: var(--text-muted);
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.message .poster-genre { color: var(--text-muted) !important; }
.message .meta-note, .message .meta-note * { color: var(--text-muted) !important; }
.meta-note {
    margin-top: var(--s3); padding: 10px var(--s3);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--bg-subtle); font-size: 12px; line-height: 1.5;
}

/* ---------- Input ---------- */
/* Gradio 6: ช่องพิมพ์ถูกห่อด้วยกล่องซ้อนกันหลายชั้น ให้เฉพาะ .styler ชั้นนอกสุดเป็นแคปซูลสีขาว */
.gradio-container .styler:has(textarea) {
    background: var(--bg) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-pill) !important;
    box-shadow: none !important;
    padding: 4px 8px !important;
    transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease) !important;
}
.gradio-container .styler:has(textarea):hover { border-color: #B8BDC6 !important; }
.gradio-container .styler:has(textarea):focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
/* กล่อง .gr-group ที่ห่ออยู่ด้านนอก (พื้นเทา/มีกรอบ) ต้องโปร่งใส */
.gradio-container .gr-group:has(textarea) {
    background: transparent !important; border: none !important; box-shadow: none !important;
    border-radius: 0 !important; padding: 0 !important;
}
/* กล่องชั้นในทั้งหมดต้องไม่มีกรอบ/พื้นหลัง */
.gradio-container .styler:has(textarea) .gr-group,
.gradio-container .styler:has(textarea) .block,
.gradio-container .styler:has(textarea) .row,
.gradio-container .styler:has(textarea) label,
.gradio-container .styler:has(textarea) .input-container {
    background: transparent !important; border: none !important; box-shadow: none !important;
    border-radius: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important;
}
.gradio-container .styler:has(textarea) textarea {
    border: none !important; background: transparent !important; box-shadow: none !important;
    border-radius: 0 !important; padding: 10px 12px !important;
}

.gradio-container textarea {
    background: var(--bg) !important; color: var(--text) !important;
    border: 1px solid var(--border-strong) !important; border-radius: var(--radius-pill) !important;
    padding: 14px 20px !important; font-size: 15px !important; box-shadow: none !important;
    transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease) !important;
}
.gradio-container textarea::placeholder { color: #9CA3AF !important; }
.gradio-container textarea:hover { border-color: #B8BDC6 !important; }
.gradio-container textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
    outline: none !important;
}

.submit-button, button.submit-button, .gradio-container button.primary {
    background: var(--text) !important; color: #FFFFFF !important;
    border: none !important; border-radius: 50% !important; box-shadow: none !important;
    width: 36px !important; height: 36px !important; min-width: 36px !important;
    transition: background var(--dur) var(--ease), opacity var(--dur) var(--ease) !important;
}
.gradio-container button.primary:hover { background: #000000 !important; }
.gradio-container button.primary:disabled { opacity: 0.35 !important; cursor: not-allowed !important; }

.gradio-container button.secondary, .gradio-container button.stop,
.gradio-container .icon-button-wrapper button {
    background: var(--bg) !important; color: var(--text-muted) !important;
    border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
    box-shadow: none !important;
    transition: background var(--dur) var(--ease), color var(--dur) var(--ease) !important;
}
.gradio-container button.secondary:hover, .gradio-container button.stop:hover {
    background: var(--bg-subtle) !important; color: var(--text) !important;
}

.examples-holder, .examples { margin-top: var(--s3) !important; }
.examples button, .example, .gradio-container .examples-holder button {
    background: var(--bg) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important;
    font-size: 13.5px !important; padding: var(--s3) var(--s4) !important;
    text-align: left !important; box-shadow: none !important;
    transition: background var(--dur) var(--ease), border-color var(--dur) var(--ease) !important;
}
.examples button:hover, .gradio-container .examples-holder button:hover {
    background: var(--bg-subtle) !important; border-color: var(--border-strong) !important;
}

/* Loading */
.gradio-container .generating, .gradio-container .pending { border-color: transparent !important; }
.gradio-container .progress-bar { background: var(--accent) !important; }
.gradio-container .loading { color: var(--text-muted) !important; }
/* สถานะกำลังตอบ (จุดกะพริบของแชต) ให้เป็นสีเทาเรียบ ๆ */
.gradio-container .pending .dot-flashing,
.gradio-container .dot-flashing,
.gradio-container .dot-flashing::before,
.gradio-container .dot-flashing::after { background-color: var(--text-muted) !important; color: var(--text-muted) !important; }
.gradio-container .generating { border: none !important; }

/* ข้อความต้องทึบเต็มที่ ไม่จางตามสถานะของ Gradio */
.gradio-container .message,
.gradio-container .message-row,
.gradio-container .bot,
.gradio-container .user,
.gradio-container .generating,
.gradio-container .pending { opacity: 1 !important; }
.gradio-container .bot, .gradio-container .bot .prose,
.gradio-container .bot .message-content,
.gradio-container .bot .message-content p { color: var(--text) !important; }

/* ชั้นโหลด/ช่องพิมพ์ตอนประมวลผล ต้องไม่เป็นแถบเทาทึบ */
.gradio-container .wrap.default,
.gradio-container .wrap.generating,
.gradio-container .wrap.translucent { background: transparent !important; }
.gradio-container textarea:disabled,
.gradio-container textarea[disabled] { background: var(--bg) !important; opacity: 0.6 !important; }

* { scrollbar-width: thin; scrollbar-color: var(--border-strong) transparent; }

/* ---------- Responsive ---------- */
@media (max-width: 1024px) {
    #sidebar { padding: var(--s4) var(--s3) !important; }
    #chat-wrapper { padding-top: var(--s4); }
}
@media (max-width: 768px) {
    #app-row { flex-wrap: wrap !important; }
    #sidebar {
        min-height: auto; width: 100% !important; position: static;
        border-right: none !important; border-bottom: 1px solid var(--border) !important;
    }
    #sidebar-inner { flex-direction: row; align-items: center; justify-content: space-between; gap: var(--s3); }
    .new-chat { padding: 8px var(--s3); }
    .page-title { font-size: 20px; }
    .gradio-container .message .poster-item { flex-basis: 92px; width: 92px !important; max-width: 92px; }
    .gradio-container .message img.poster-img {
        width: 92px !important; min-width: 92px !important; max-width: 92px !important;
        height: 136px !important; min-height: 136px !important; max-height: 136px !important;
    }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
"""

# ---------------------------------------------------------------------------
# HTML ส่วนคงที่
# ---------------------------------------------------------------------------
ICON_FILM = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="3" y="4" width="18" height="16" rx="2"/>'
    '<path d="M7 4v16M17 4v16M3 9h4M17 9h4M3 15h4M17 15h4"/></svg>'
)
ICON_PLUS = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>'
)

SIDE = f"""
<div id="sidebar-inner">
  <div class="brand">{ICON_FILM}<span>Movie Assistant</span></div>

  <a class="new-chat" href="/" aria-label="เริ่มแชทใหม่">{ICON_PLUS}<span>แชทใหม่</span></a>
</div>
"""

HERO = """
<div>
  <h1 class="page-title">วันนี้อยากดูหนังแนวไหน</h1>
  <p class="page-subtitle">ค้นหนังตามความหมาย (RAG) แล้วให้ AI ในเครื่องเรียบเรียงคำแนะนำเป็นภาษาไทย</p>
</div>
"""

EMPTY_STATE = """
<div class="empty-state">
  <div class="empty-title">อยากดูหนังแบบไหนดี?</div>
  <div class="empty-text">กดตัวอย่างด้านล่าง หรือพิมพ์เองได้เลย</div>
</div>
"""

# ---------------------------------------------------------------------------
# ฟังก์ชันแชต (ลอจิกเดิม)
# ---------------------------------------------------------------------------
MARK = "\n\n---\n"
_poster_cache = {}


def genre_text(m):
    """ประเภทหนังจากข้อมูลเดิม (ถ้ามีฟิลด์ genres/genre) ไม่มีข้อมูลจะคืนค่าว่าง ไม่สร้างข้อมูลขึ้นเอง"""
    g = m.get("genres") or m.get("genre")
    if not g:
        return ""
    if isinstance(g, (list, tuple)):
        g = ", ".join(str(x) for x in g)
    return html.escape(str(g).replace("|", ", "))


def poster_card(m):
    """การ์ดโปสเตอร์ของหนังเรื่องนั้น (ฝังเป็น base64 ใช้ออฟไลน์ได้) ไม่มีไฟล์ภาพจะคืนค่าว่าง"""
    path = os.path.join("posters", f"{m['id']}.jpg")
    if not os.path.exists(path):
        return ""
    if path not in _poster_cache:
        with open(path, "rb") as f:
            _poster_cache[path] = base64.b64encode(f.read()).decode()
    title = html.escape(str(m["title"]))
    year = html.escape(str(m["year"]))
    genre = genre_text(m)
    genre_html = f'<div class="poster-genre">{genre}</div>' if genre else ""
    return (
        '<div class="poster-item">'
        f'<img src="data:image/jpeg;base64,{_poster_cache[path]}" class="poster-img" '
        f'alt="โปสเตอร์ {title}" loading="lazy">'
        f'<div class="poster-title">{title} ({year})</div>{genre_html}</div>'
    )


def to_text(content):
    """Gradio รุ่นใหม่อาจส่งข้อความมาเป็นลิสต์ ให้แปลงเป็นข้อความธรรมดา"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def chat(message, history):
    message = to_text(message)
    # ตัดส่วน "หนังที่ค้นเจอ" ออกจากประวัติ ไม่ให้ AI สับสน
    past = [{"role": h["role"], "content": to_text(h["content"]).split(MARK)[0]} for h in history][-6:]

    # Error state: แจ้งผู้ใช้อย่างสุภาพเมื่อระบบเบื้องหลังมีปัญหา
    try:
        # คำตอบ + หนังที่ AI อ่านจริง + หนังที่น่าจะหมายถึง มาจากการค้นรอบเดียวกัน (โปสเตอร์จึงตรงกับคำตอบ)
        answer, candidates, suggestions = recommend_with_candidates(message, past)
    except Exception as exc:  # noqa: BLE001
        return (
            '<div class="error-box"><b>ขออภัย เกิดข้อผิดพลาด</b><br>'
            "ไม่สามารถประมวลผลคำขอได้ในตอนนี้ กรุณาตรวจสอบว่า Ollama ทำงานอยู่ แล้วลองอีกครั้ง"
            f"<br><small>{html.escape(str(exc))}</small></div>"
        )

    # โปสเตอร์ของเรื่องที่ AI พูดถึงในคำตอบ ถ้าไม่เจอชื่อเลยให้แสดง 3 เรื่องแรกที่ค้นเจอ
    # โปสเตอร์เฉพาะเรื่องที่ AI พูดถึงในคำตอบ ถ้าไม่มีเลยใช้เฉพาะหนังที่ชื่อตรงกับที่ผู้ใช้พิมพ์/ทาย (ไม่เอาผลค้นสุ่มมาโชว์)
    picked = [m for m in candidates if m["title"].lower() in answer.lower()] or suggestions[:3]
    cards = "".join(poster_card(m) for m in picked[:4])
    posters = f'<div class="poster-grid">{cards}</div>' if cards else ""
    found = " · ".join(f"{html.escape(str(m['title']))} ({html.escape(str(m['year']))})" for m in candidates)
    meta = f'<div class="meta-note">หนังที่ระบบค้นเจอก่อนให้ AI เลือก: {found}</div>' if found else ""
    guess = " · ".join(f"{html.escape(str(m['title']))} ({html.escape(str(m['year']))})" for m in suggestions)
    hint = f'<div class="meta-note">หนังที่น่าจะหมายถึง: {guess}</div>' if guess else ""
    return f"{answer}{MARK}{posters}{hint}{meta}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# Gradio บางรุ่นรับ theme/css ที่ launch() บางรุ่นรับที่ Blocks() เช็กให้อัตโนมัติ
STYLE_AT_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters
style = {"theme": THEME, "css": CSS}

with gr.Blocks(title="Movie Assistant", **({} if STYLE_AT_LAUNCH else style)) as demo:
    with gr.Row(equal_height=False, elem_id="app-row"):
        with gr.Column(scale=2, min_width=220, elem_id="sidebar"):
            gr.HTML(SIDE)
        with gr.Column(scale=9, elem_id="main"):
            with gr.Column(elem_id="chat-wrapper"):
                gr.HTML(HERO)
                gr.ChatInterface(
                    fn=chat,
                    chatbot=gr.Chatbot(
                        height="60vh",
                        sanitize_html=False,
                        show_label=False,
                        placeholder=EMPTY_STATE,
                    ),
                    textbox=gr.Textbox(
                        placeholder="พิมพ์ข้อความ เช่น หนังตลกดูกับครอบครัว",
                        container=False,
                        scale=7,
                        submit_btn=True,
                        stop_btn=True,
                    ),
                    examples=[
                        "อยากดูหนังรักโรแมนติกเบา ๆ ดูแล้วอารมณ์ดี",
                        "หนังระทึกขวัญหักมุม ไม่โหดมาก",
                        "หนังแอนิเมชันดูแล้วอบอุ่นใจ",
                        "หนังไซไฟที่ทำให้ต้องคิดตาม",
                    ],
                )

if __name__ == "__main__":
    demo.launch(**(style if STYLE_AT_LAUNCH else {}))