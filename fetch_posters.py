"""ดาวน์โหลดโปสเตอร์หนังทุกเรื่องใน movies.json เก็บไว้ในโฟลเดอร์ posters/

จับคู่ภาพจากรหัส tmdbId ใน links.csv ของ MovieLens จึงตรงเรื่องแน่นอน
ต้องมีเน็ตและ TMDB API key เฉพาะตอนรันสคริปต์นี้ครั้งเดียว หลังจากนั้นใช้ออฟไลน์ได้

รัน:  python fetch_posters.py   (ถ้ายังไม่ได้ตั้งค่า key จะถามให้พิมพ์)
ใช้ "API Key" แบบสั้น (v3) ไม่ใช่ Read Access Token
"""
import csv
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests

API_KEY = os.environ.get("TMDB_API_KEY") or input("ใส่ TMDB API key: ").strip()
ZIP = "ml-latest-small.zip"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
OUT = "posters"
IMG = "https://image.tmdb.org/t/p/w342"


def load_links():
    data = open(ZIP, "rb").read() if os.path.exists(ZIP) else requests.get(ZIP_URL, timeout=60).content
    z = zipfile.ZipFile(io.BytesIO(data))
    with z.open("ml-latest-small/links.csv") as f:
        rows = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
        return {int(r["movieId"]): r["tmdbId"] for r in rows if r["tmdbId"]}


def fetch(task):
    movie_id, tmdb_id = task
    path = os.path.join(OUT, f"{movie_id}.jpg")
    if os.path.exists(path):
        return "skip"
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}", params={"api_key": API_KEY}, timeout=15
        )
        if r.status_code == 401:
            return "auth"
        r.raise_for_status()
        poster = r.json().get("poster_path")
        if not poster:
            return "none"
        img = requests.get(IMG + poster, timeout=30)
        img.raise_for_status()
        with open(path, "wb") as f:
            f.write(img.content)
        return "ok"
    except Exception:
        return "error"


def main():
    os.makedirs(OUT, exist_ok=True)
    with open("movies.json", encoding="utf-8") as f:
        movies = json.load(f)
    links = load_links()
    tasks = [(m["id"], links[m["id"]]) for m in movies if m["id"] in links]
    print(f"หนัง {len(movies)} เรื่อง มีรหัส TMDB {len(tasks)} เรื่อง กำลังโหลดโปสเตอร์...")

    counts = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, result in enumerate(pool.map(fetch, tasks), 1):
            counts[result] = counts.get(result, 0) + 1
            if result == "auth":
                raise SystemExit("API key ไม่ถูกต้อง (401) ตรวจว่าใช้ API Key แบบ v3")
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)}")

    print(f"เสร็จแล้ว: โหลดใหม่ {counts.get('ok', 0)}, มีอยู่แล้ว {counts.get('skip', 0)}, "
          f"ไม่มีภาพ {counts.get('none', 0)}, ผิดพลาด {counts.get('error', 0)}")
    if counts.get("error"):
        print("มีบางเรื่องผิดพลาด ลองรันสคริปต์ซ้ำ จะโหลดเฉพาะที่ยังขาด")


if __name__ == "__main__":
    main()
