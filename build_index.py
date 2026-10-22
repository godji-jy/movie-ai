"""ขั้นที่ 2: แปลงข้อมูลหนังเป็นเวกเตอร์ แล้วบันทึกเป็น embeddings.npy

ใช้โมเดล multilingual-e5-small (เล็ก รองรับภาษาไทย รันบน CPU ได้สบาย
จะได้ไม่แย่ง VRAM กับ LLM)
ถ้าในข้อมูลมีคำโปรย (tagline) และคำสำคัญ (keywords) จะนำมารวมในการค้นด้วย
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/multilingual-e5-small"


def movie_to_text(m):
    genres = ", ".join(m["genres"])
    parts = [f"{m['title']} ({m['original_title']}, {m['year']}) แนว: {genres}."]
    if m.get("tagline"):
        parts.append(f"คำโปรย: {m['tagline']}.")
    parts.append(m["overview"])
    if m.get("keywords"):
        parts.append("คำสำคัญ: " + ", ".join(m["keywords"]) + ".")
    return " ".join(parts)


def main():
    with open("movies.json", encoding="utf-8") as f:
        movies = json.load(f)

    model = SentenceTransformer(EMBED_MODEL)
    # e5 ต้องใส่คำนำหน้า "passage: " กับข้อความที่จะเก็บ
    texts = ["passage: " + movie_to_text(m) for m in movies]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    np.save("embeddings.npy", vectors)
    print(f"สร้าง embeddings ของ {len(movies)} เรื่องเรียบร้อย")


if __name__ == "__main__":
    main()
