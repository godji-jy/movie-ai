"""ขั้นที่ 1: ดึงข้อมูลหนังจาก TMDB แล้วบันทึกเป็น movies.json

ก่อนรัน ตั้งค่า API key (สมัครฟรีที่ themoviedb.org -> Settings -> API):
  PowerShell:  $env:TMDB_API_KEY="ใส่key"
  CMD:         set TMDB_API_KEY=ใส่key
"""
import json
import os
import time

import requests

API_KEY = os.environ.get("TMDB_API_KEY")
BASE = "https://api.themoviedb.org/3"
PAGES = 25  # หน้าละ 20 เรื่อง ต่อ 1 ประเภทรายการ


def get(path, **params):
    params["api_key"] = API_KEY
    r = requests.get(f"{BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    if not API_KEY:
        raise SystemExit("ยังไม่ได้ตั้งค่า TMDB_API_KEY")

    genres = {g["id"]: g["name"] for g in get("/genre/movie/list", language="th-TH")["genres"]}
    movies = {}

    for category in ("popular", "top_rated"):
        for page in range(1, PAGES + 1):
            th = get(f"/movie/{category}", language="th-TH", page=page)["results"]
            en = {m["id"]: m for m in get(f"/movie/{category}", language="en-US", page=page)["results"]}

            for m in th:
                overview = m.get("overview") or en.get(m["id"], {}).get("overview", "")
                if not overview:
                    continue
                movies[m["id"]] = {
                    "id": m["id"],
                    "title": m["title"],
                    "original_title": m["original_title"],
                    "year": (m.get("release_date") or "")[:4],
                    "genres": [genres[g] for g in m.get("genre_ids", []) if g in genres],
                    "overview": overview,
                    "rating": m.get("vote_average"),
                }
            time.sleep(0.1)
            print(f"{category} หน้า {page}/{PAGES} (รวม {len(movies)} เรื่อง)")

    with open("movies.json", "w", encoding="utf-8") as f:
        json.dump(list(movies.values()), f, ensure_ascii=False, indent=2)
    print(f"บันทึก {len(movies)} เรื่องลง movies.json แล้ว")


if __name__ == "__main__":
    main()