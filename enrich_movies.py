"""เพิ่มคำโปรย (tagline) และคำสำคัญ (keywords) จาก TMDB ให้หนังทุกเรื่องใน movies.json

ใช้ได้เฉพาะ movies.json ที่สร้างจาก tmdb_movies.py (id ต้องเป็นรหัส TMDB)
ต้องมีเน็ตและ TMDB API key ตอนรัน หลังรันเสร็จให้รัน python build_index.py ต่อ
รันซ้ำได้ จะข้ามเรื่องที่เติมข้อมูลแล้ว
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_KEY = os.environ.get("TMDB_API_KEY") or input("ใส่ TMDB API key: ").strip()
BASE = "https://api.themoviedb.org/3"
MAX_KEYWORDS = 15


def fetch(m):
    if "keywords" in m:
        return "skip"
    for attempt in range(3):
        try:
            r = requests.get(
                f"{BASE}/movie/{m['id']}",
                params={"api_key": API_KEY, "language": "th-TH", "append_to_response": "keywords"},
                timeout=15,
            )
            if r.status_code == 401:
                return "auth"
            if r.status_code == 404:
                m["tagline"], m["keywords"] = "", []
                return "ok"
            r.raise_for_status()
            d = r.json()
            m["tagline"] = (d.get("tagline") or "").strip()
            m["keywords"] = [k["name"] for k in d.get("keywords", {}).get("keywords", [])][:MAX_KEYWORDS]
            return "ok"
        except requests.RequestException:
            time.sleep(1 + attempt)
    return "error"


def main():
    if not os.path.exists(os.path.join("posters", ".tmdb")):
        raise SystemExit("movies.json ไม่ได้มาจาก tmdb_movies.py (รหัสหนังจะไม่ตรงกับ TMDB) จึงเพิ่มข้อมูลไม่ได้")

    with open("movies.json", encoding="utf-8") as f:
        movies = json.load(f)
    print(f"กำลังเพิ่มคำโปรยและคำสำคัญให้หนัง {len(movies)} เรื่อง...")

    counts = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(fetch, movies), 1):
            counts[result] = counts.get(result, 0) + 1
            if result == "auth":
                raise SystemExit("API key ไม่ถูกต้อง (401) ตรวจว่าใช้ API Key แบบ v3")
            if i % 300 == 0:
                print(f"  {i}/{len(movies)}")

    with open("movies.json", "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
    print(f"เสร็จแล้ว: เพิ่มใหม่ {counts.get('ok', 0)}, มีอยู่แล้ว {counts.get('skip', 0)}, ผิดพลาด {counts.get('error', 0)}")
    if counts.get("error"):
        print("มีบางเรื่องผิดพลาด ลองรันซ้ำ จะเติมเฉพาะเรื่องที่ยังขาด")
    print("ขั้นต่อไป: python build_index.py")


if __name__ == "__main__":
    main()
