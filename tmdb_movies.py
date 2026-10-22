"""ดึงหนังจาก TMDB เป็นพัน ๆ เรื่อง (เรียงตามจำนวนคนโหวต) พร้อมเรื่องย่อภาษาไทยและโปสเตอร์

ผลลัพธ์: movies.json (ใช้แทนของเดิม) และโฟลเดอร์ posters/ (ชื่อไฟล์ = id หนัง)
ต้องมีเน็ตและ TMDB API key ตอนรันครั้งเดียว หลังจากนั้นใช้ออฟไลน์ได้

รัน:  python tmdb_movies.py   แล้ว   python build_index.py
"""
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_KEY = os.environ.get("TMDB_API_KEY") or input("ใส่ TMDB API key: ").strip()
PAGES = 1000  # หน้าละ 20 เรื่อง: 150 หน้า = ราว 3,000 เรื่อง (สูงสุด 500 หน้า)
THAI_PAGES = 60  # หนังไทยเพิ่ม: 60 หน้า = ราว 1,200 เรื่อง (ถ้ามีไม่ถึง จะได้เท่าที่มี)
BASE = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p/w342"
OUT = "posters"


def get(path, **params):
    params["api_key"] = API_KEY
    for attempt in range(3):
        try:
            r = requests.get(BASE + path, params=params, timeout=15)
            if r.status_code == 401:
                raise SystemExit("API key ไม่ถูกต้อง (401) ตรวจว่าใช้ API Key แบบ v3")
            if r.status_code == 422:  # หน้าเกินจำนวนที่มี
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(1 + attempt)
    return None


def fetch_page(task):
    page, extra = task
    q = dict(sort_by="vote_count.desc", include_adult="false", page=page, **extra)
    th = get("/discover/movie", language="th-TH", **q)
    en = get("/discover/movie", language="en-US", **q)
    if not th or not en:
        return []
    en_map = {m["id"]: m for m in en["results"]}
    rows = []
    for m in th["results"]:
        e = en_map.get(m["id"], {})
        overview = m.get("overview") or e.get("overview") or ""
        poster = m.get("poster_path") or e.get("poster_path")
        if not overview or not poster:
            continue
        rows.append({
            "id": m["id"],
            "title": m["title"],
            "original_title": m["original_title"],
            "year": (m.get("release_date") or "")[:4],
            "genre_ids": m.get("genre_ids", []),
            "overview": overview,
            "rating": m.get("vote_average") or 0,
            "poster": poster,
        })
    return rows


def download(movie):
    path = os.path.join(OUT, f"{movie['id']}.jpg")
    if os.path.exists(path):
        return True
    try:
        r = requests.get(IMG + movie["poster"], timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except requests.RequestException:
        return False


def prepare_folders():
    # สำรองไฟล์เดิม (MovieLens) และโฟลเดอร์ภาพเดิม กันรหัสหนังชนกันจนภาพผิดเรื่อง
    #if os.path.exists("movies.json") and not os.path.exists("movies_movielens.json"):
       # shutil.copy("movies.json", "movies_movielens.json")
    marker = os.path.join(OUT, ".tmdb")
    if os.path.isdir(OUT) and not os.path.exists(marker):
        backup, n = "posters_old", 1
        while os.path.exists(backup):
            n += 1
            backup = f"posters_old{n}"
        os.rename(OUT, backup)
        print(f"ย้ายโฟลเดอร์ภาพเดิมไปเก็บที่ {backup}")
    os.makedirs(OUT, exist_ok=True)
    open(marker, "w").close()


def main():
    prepare_folders()
    genres = {g["id"]: g["name"] for g in get("/genre/movie/list", language="th-TH")["genres"]}

    pages = min(PAGES, 1000)
    print(f"กำลังดึงรายชื่อหนัง {pages} หน้า + หนังไทย {THAI_PAGES} หน้า...")
    tasks = [(pg, {}) for pg in range(1, pages + 1)]
    tasks += [(pg, {"with_original_language": "th"}) for pg in range(1, THAI_PAGES + 1)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch_page, tasks))

    movies = {}
    for rows in results:
        for m in rows:
            movies[m["id"]] = m
    movies = list(movies.values())
    print(f"ได้หนัง {len(movies)} เรื่อง กำลังโหลดโปสเตอร์ (ใช้เวลาสักครู่)...")

    with ThreadPoolExecutor(max_workers=8) as pool:
        ok = list(pool.map(download, movies))

    for m, got in zip(movies, ok):
        m["genres"] = [genres[g] for g in m.pop("genre_ids") if g in genres]
        m.pop("poster")
        m["has_poster"] = got

    with open("movies.json", "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
    print(f"บันทึก {len(movies)} เรื่องลง movies.json แล้ว (โปสเตอร์โหลดสำเร็จ {sum(ok)} เรื่อง)")
    print("ขั้นต่อไป: python build_index.py")


if __name__ == "__main__":
    main()