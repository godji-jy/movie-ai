"""ตัวหลัก: ค้นหนังด้วย AI ตัวที่ 1 (embedding) แล้วให้ AI ตัวที่ 2 (LLM) เรียบเรียงคำแนะนำ"""
import json
import re
import threading

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/multilingual-e5-small"
LLM_NAME = "scb10x/typhoon2.5-qwen3-4b"  # โมเดลหลัก (ตัวเดียวที่ใช้) ติดตั้งด้วย: ollama pull scb10x/typhoon2.5-qwen3-4b
FAST_LLM = "scb10x/typhoon2.5-qwen3-4b"  # ตัวสำรองที่ลองได้: "scb10x/typhoon2.1-gemma3-4b" หรือ "qwen3:4b"
LLM_CANDIDATES = 5         # จำนวนหนังที่ส่งให้ LLM อ่าน (น้อยลง = อ่านเร็วขึ้น)
OVERVIEW_CHARS = 160       # ตัดเรื่องย่อให้สั้นลงก่อนส่งให้ LLM

SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยแนะนำหนังที่เป็นมิตร ตอบเป็นภาษาไทย "
    "แนะนำเฉพาะหนังที่อยู่ในรายการที่ให้มาเท่านั้น ห้ามแต่งชื่อหนังหรือเรื่องย่อเพิ่มเอง "
    "เลือกมา 3 เรื่องที่เหมาะที่สุด พร้อมเหตุผลสั้น ๆ ว่าทำไมตรงกับที่ผู้ใช้ต้องการ "
    "ถ้าไม่มีเรื่องไหนเหมาะเลย ให้บอกตรง ๆ "
    "ถ้าผู้ใช้พิมพ์ชื่อหนังผิด หรือจำชื่อไม่ได้และเล่าเรื่องคร่าว ๆ มา ให้เดาจากรายการว่าน่าจะหมายถึงเรื่องไหน "
    "แล้วบอกว่า 'น่าจะเป็นเรื่อง ...' ถ้าไม่แน่ใจให้บอกว่าไม่แน่ใจ"
)

_embedder = SentenceTransformer(EMBED_MODEL)
_vectors = np.load("embeddings.npy")
with open("movies.json", encoding="utf-8") as f:
    _movies = json.load(f)


def search(query, k=8):
    q = _embedder.encode(["query: " + query], normalize_embeddings=True)[0]
    scores = _vectors @ q
    top = np.argsort(-scores)[:k]
    return [_movies[i] for i in top]


def format_movies(movies, limit=300):
    lines = []
    for i, m in enumerate(movies, 1):
        overview = m["overview"][:limit]
        lines.append(
            f"{i}. {m['title']} ({m['year']}) | แนว: {', '.join(m['genres'])} | "
            f"คะแนน: {m['rating']:.1f}\n   เรื่องย่อ: {overview}"
        )
    return "\n".join(lines)


def _messages(query, candidates, history, limit):
    user_msg = f"รายการหนังที่เลือกได้:\n{format_movies(candidates, limit)}\n\nคำขอของผู้ใช้: {query}"
    return [{"role": "system", "content": SYSTEM_PROMPT}] + (history or []) + [{"role": "user", "content": user_msg}]


# ---------------------------------------------------------------------------
# ตัวช่วยสำหรับ Qwen3: ปิดโหมดคิด (thinking) เพื่อให้ตอบเร็วและไม่มี <think> หลุดมา
# ---------------------------------------------------------------------------
def _chat(messages, **kwargs):
    """เรียก ollama.chat พร้อม think=False ถ้าไลบรารีรุ่นเก่าไม่รู้จัก ให้ใช้ /no_think แทน"""
    try:
        return ollama.chat(messages=messages, think=False, **kwargs)
    except TypeError:
        messages = [dict(m) for m in messages]
        messages[-1]["content"] += " /no_think"
        return ollama.chat(messages=messages, **kwargs)


def _strip_think(text):
    """ตัดส่วนความคิด <think>...</think> ออก (กันกรณีโมเดลยังพ่นมา)"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def stream_answer(query, candidates, history=None, model=LLM_NAME):
    """ให้ LLM ตอบทีละคำ (เห็นคำตอบเร็วขึ้นโดยไม่ต้องรอจนจบ)"""
    stream = _chat(
        _messages(query, candidates, history, OVERVIEW_CHARS),
        model=model,
        stream=True,
        keep_alive="30m",  # ให้โมเดลค้างในหน่วยความจำ ไม่ต้องโหลดใหม่ทุกคำถาม
        options={"temperature": 0.6, "num_predict": 400, "num_ctx": 3072},
    )
    in_think = False  # ข้ามข้อความที่อยู่ในบล็อก <think>...</think> ระหว่างสตรีม
    for chunk in stream:
        text = chunk["message"]["content"]
        if "<think>" in text:
            in_think = True
        if in_think:
            if "</think>" in text:
                in_think = False
                text = text.split("</think>", 1)[1].lstrip()
            else:
                continue
        if text:
            yield text


def recommend(query, history=None):
    """แบบไม่สตรีม (ใช้กับ main.py)"""
    return recommend_with_candidates(query, history)[0]


def warm_up_async(model=LLM_NAME):
    """โหลดโมเดลเข้าหน่วยความจำล่วงหน้าตอนเปิดโปรแกรม คำถามแรกจะไม่ช้า"""
    def _run():
        try:
            ollama.generate(model=model, prompt="", keep_alive="30m")
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# แก้คำผิด / เดาชื่อเรื่อง: จับคู่ชื่อหนังแบบ "ใกล้เคียง" (ไม่ต้องใช้โมเดลเพิ่ม เร็วมาก)
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz, process
except ImportError:  # ไม่มี rapidfuzz ใช้ difflib ที่มากับ Python แทน (ช้ากว่าเล็กน้อย)
    fuzz = process = None
    import difflib


def _norm(text):
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


_title_keys = []  # (ชื่อที่ปรับรูปแล้ว, ลำดับหนังใน _movies)
for _i, _m in enumerate(_movies):
    for _t in {_m["title"], _m["original_title"]}:
        _key = _norm(_t)
        if _key:
            _title_keys.append((_key, _i))
_choices = [k for k, _ in _title_keys]


def title_matches(query, limit=2, cutoff=80):
    """หาหนังที่ชื่อใกล้เคียงกับที่พิมพ์ (พิมพ์ผิดนิดหน่อยก็เจอ) ใช้กับข้อความสั้น ๆ เท่านั้น"""
    q = _norm(query)
    if not q or len(str(query)) > 40:
        return []
    if process is not None:
        hits = [(score, idx) for _, score, idx in
                process.extract(q, _choices, scorer=fuzz.ratio, limit=limit * 4, score_cutoff=cutoff)]
    else:
        close = difflib.get_close_matches(q, _choices, n=limit * 4, cutoff=cutoff / 100)
        hits = [(difflib.SequenceMatcher(None, q, c).ratio() * 100, _choices.index(c)) for c in close]

    result, seen = [], set()
    for _, idx in sorted(hits, key=lambda h: -h[0]):
        mi = _title_keys[idx][1]
        if mi not in seen:
            seen.add(mi)
            result.append(_movies[mi])
        if len(result) >= limit:
            break
    return result


GUESS_LLM = LLM_NAME      # โมเดลที่ใช้ "ทายชื่อหนังจากความจำ" (โมเดลใหญ่ขึ้นจะรู้จักหนังมากขึ้น)
GUESS_MIN_CHARS = 20      # ข้อความสั้นกว่านี้ถือว่าเป็นชื่อ ไม่ต้องทาย
GUESS_PROMPT = (
    "ผู้ใช้จำชื่อหนังไม่ได้ จึงเล่าเนื้อเรื่อง ตัวละคร นักแสดง หรือคำพูดในเรื่องมา "
    "จงนึกถึงหนังที่ตรงกับคำบรรยายมากที่สุด แล้วตอบเป็น JSON รูปแบบ "
    '{"titles": ["ชื่อหนังภาษาอังกฤษ", ...]} สูงสุด 3 เรื่อง เรียงจากที่น่าจะใช่ที่สุด '
    'ถ้าไม่รู้จักหรือไม่แน่ใจ ให้ตอบ {"titles": []} ห้ามแต่งชื่อหนังที่ไม่มีอยู่จริง'
)


def guess_titles(query):
    """ให้ LLM ทายชื่อหนังจากคำบรรยาย (เช่น "ฮีโร่ไม่ใช่คนดี โดนด่าว่าไอ้ถ่อย" -> Hancock)
    ผลที่ได้ยังไม่ถือว่าจริง ต้องไปจับคู่กับรายชื่อหนังในฐานข้อมูลอีกที"""
    try:
        res = _chat(
            [{"role": "system", "content": GUESS_PROMPT}, {"role": "user", "content": query}],
            model=GUESS_LLM,
            format="json",
            keep_alive="30m",
            options={"temperature": 0.2, "num_predict": 100, "num_ctx": 2048},
        )
        text = _strip_think(res["message"]["content"])
        data = json.loads(text)
        titles = data.get("titles", []) if isinstance(data, dict) else data
        return [str(t) for t in titles if isinstance(t, str)][:3]
    except Exception:
        return []  # ทายไม่ได้ก็ข้ามไป ใช้การค้นตามความหมายตามปกติ


def smart_search(query, k=5, use_guess=True):
    """คืน (หนังที่ส่งให้ AI, หนังที่น่าจะหมายถึง)
    หนังที่ชื่อใกล้เคียง หรือที่ AI ทายแล้วเจอในฐานข้อมูล จะถูกวางไว้หน้าสุด
    ส่วนที่เหลือมาจากการค้นตามความหมาย (เล่าเรื่อง/บรรยายก็เจอ)"""
    suggestions = title_matches(query)
    if use_guess and not suggestions and len(str(query)) >= GUESS_MIN_CHARS:
        for t in guess_titles(query):
            for m in title_matches(t, limit=1, cutoff=88):  # ยืนยันกับฐานข้อมูลจริงเท่านั้น
                if m["id"] not in {s["id"] for s in suggestions}:
                    suggestions.append(m)
    seen = {m["id"] for m in suggestions}
    rest = [m for m in search(query, k=k + len(suggestions)) if m["id"] not in seen]
    return (suggestions + rest)[:k], suggestions


# ---------------------------------------------------------------------------
# ถามตอบต่อเนื่อง: เข้าใจคำถามต่อยอด เช่น "ไม่ใช่เรื่องนี้" "เรื่องแรกใครเล่น" "มีอีกไหม"
# ---------------------------------------------------------------------------
REJECT_WORDS = ("ไม่ใช่", "ไม่ถูก", "ผิด", "ไม่เจอ", "ไม่ตรง", "เรื่องอื่น", "อีกเรื่อง", "อันอื่น", "nope", "wrong")

CHAT_PROMPT = (
    "คุณคือผู้ช่วยแนะนำหนังที่เป็นมิตร ตอบเป็นภาษาไทย กำลังคุยต่อเนื่องกับผู้ใช้ "
    "ตอบจากข้อมูลหนังที่ให้มาและบทสนทนาก่อนหน้าเท่านั้น ห้ามแต่งข้อมูลที่ไม่มี "
    "ถ้าไม่มีข้อมูลที่ถูกถาม ให้บอกว่าไม่ทราบจากข้อมูลที่มี"
)

PLAN_PROMPT = (
    "คุณช่วยวางแผนการตอบในบทสนทนาแนะนำหนัง ดูประวัติและข้อความล่าสุดของผู้ใช้ แล้วตอบเป็น JSON รูปแบบ "
    '{"mode": "search" หรือ "chat", "query": "...", "exclude": ["..."]} โดย\n'
    '- mode "search" = ต้องค้นหนังใหม่ (ผู้ใช้ขอหนัง, เล่าเรื่องเพิ่ม, หรือบอกว่าเรื่องที่แนะนำไม่ใช่)\n'
    '- mode "chat" = ไม่ต้องค้นใหม่ (ถามรายละเอียดของหนังที่เพิ่งคุยกัน, ทักทาย, ขอบคุณ)\n'
    '- query = คำค้นหนังที่สมบูรณ์ในตัวเอง รวมรายละเอียดจากข้อความก่อนหน้าที่ยังใช้ได้ (เช่น เนื้อเรื่องที่ผู้ใช้เล่าไว้)\n'
    '- exclude = ชื่อหนังที่ผู้ใช้ปฏิเสธหรือบอกว่าไม่ใช่ (ถ้าไม่มีให้ [])'
)


def _mentioned_movies(text, limit=8):
    """หนังในฐานข้อมูลที่ถูกพูดถึงในข้อความ (ใช้ดูว่าก่อนหน้านี้แนะนำเรื่องอะไรไปแล้ว)"""
    low = str(text).lower()
    found = [m for m in _movies if len(m["title"]) >= 4 and m["title"].lower() in low]
    found.sort(key=lambda m: low.find(m["title"].lower()))
    return found[:limit]


def _last_assistant(history):
    for h in reversed(history or []):
        if h.get("role") == "assistant":
            return str(h.get("content", ""))
    return ""


def _last_user(history):
    for h in reversed(history or []):
        if h.get("role") == "user":
            # ข้อความผู้ใช้ในประวัติมีรายการหนังแนบอยู่ ตัดเอาเฉพาะคำขอจริง
            return str(h.get("content", "")).split("คำขอของผู้ใช้:")[-1].strip()
    return ""


def plan_turn(query, history):
    """ตัดสินว่าต้องค้นหนังใหม่หรือแค่คุยต่อ พร้อมสร้างคำค้นที่สมบูรณ์ในตัวเอง"""
    last_ai = _last_assistant(history)
    rejected = len(query) <= 40 and any(w in query.lower() for w in REJECT_WORDS)
    plan = {"mode": "search", "query": query, "exclude": []}

    if history:
        recent = "\n".join(
            f"{'ผู้ใช้' if h.get('role') == 'user' else 'ผู้ช่วย'}: "
            f"{str(h.get('content', '')).split('คำขอของผู้ใช้:')[-1].strip()[:400]}"
            for h in history[-4:]
        )
        try:
            res = _chat(
                [{"role": "system", "content": PLAN_PROMPT},
                 {"role": "user", "content": f"ประวัติ:\n{recent}\n\nข้อความล่าสุดของผู้ใช้: {query}"}],
                model=GUESS_LLM,
                format="json",
                keep_alive="30m",
                options={"temperature": 0.1, "num_predict": 150, "num_ctx": 2048},
            )
            data = json.loads(_strip_think(res["message"]["content"]))
            if data.get("mode") in ("search", "chat"):
                plan["mode"] = data["mode"]
            if isinstance(data.get("query"), str) and data["query"].strip():
                plan["query"] = data["query"].strip()
            if isinstance(data.get("exclude"), list):
                plan["exclude"] = [str(t) for t in data["exclude"]]
        except Exception:
            pass  # ถ้าวางแผนไม่ได้ ใช้กติกาสำรองด้านล่าง

    if rejected:
        plan["mode"] = "search"  # ผู้ใช้บอกว่าไม่ใช่ ต้องหาเรื่องอื่นเสมอ
        # คงคำค้นเดิมของผู้ใช้ไว้ ถ้าโมเดลสร้างคำค้นมาไม่ได้เรื่อง (สั้นหรือเท่ากับคำปฏิเสธ)
        if plan["query"] == query or len(plan["query"]) < 8:
            plan["query"] = _last_user(history) or query
    elif history and len(query) < 15 and plan["query"] == query and plan["mode"] == "search":
        plan["query"] = f"{_last_user(history)} {query}".strip()  # ข้อความสั้นมาก ต่อจากคำขอก่อนหน้า

    plan["exclude_ids"] = {m["id"] for m in _mentioned_movies(last_ai)} if rejected else set()
    for t in plan["exclude"]:
        for m in title_matches(t, limit=1, cutoff=88):
            plan["exclude_ids"].add(m["id"])
    return plan


def recommend_with_candidates(query, history=None, k=8):
    """ถามตอบแบบมีบริบท คืน (คำตอบ, หนังที่ AI อ่าน, หนังที่น่าจะหมายถึง)
    หน้าจอควรใช้ผลลัพธ์ชุดเดียวกันนี้แสดงโปสเตอร์ เพื่อให้ตรงกับที่ AI ตอบ"""
    plan = plan_turn(query, history)
    suggestions = []

    if plan["mode"] == "chat":
        # คุยต่อ: ใช้หนังที่เพิ่งพูดถึงเป็นข้อมูล ไม่ค้นใหม่
        recent_text = " ".join(str(h.get("content", "")) for h in (history or [])[-4:])
        candidates = _mentioned_movies(recent_text, limit=k)
        system = CHAT_PROMPT
        user_msg = f"ข้อมูลหนังที่เกี่ยวข้อง:\n{format_movies(candidates, 300)}\n\nข้อความของผู้ใช้: {query}"
    else:
        found, suggestions = smart_search(plan["query"], k=k + len(plan["exclude_ids"]))
        exclude = plan["exclude_ids"]
        candidates = [m for m in found if m["id"] not in exclude][:k]
        suggestions = [m for m in suggestions if m["id"] not in exclude]
        system = SYSTEM_PROMPT
        extra = ""
        if exclude:
            names = ", ".join(m["title"] for m in _movies if m["id"] in exclude)
            extra = f"\n(ผู้ใช้บอกว่าไม่ใช่เรื่อง: {names} ห้ามแนะนำซ้ำ)"
        user_msg = (
            f"รายการหนังที่เลือกได้:\n{format_movies(candidates, 300)}\n\n"
            f"คำขอของผู้ใช้: {plan['query']}{extra}"
        )

    messages = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": user_msg}]
    res = _chat(messages, model=LLM_NAME, options={"temperature": 0.6, "num_ctx": 4096})
    return _strip_think(res["message"]["content"]), candidates, suggestions