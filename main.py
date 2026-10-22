"""ขั้นที่ 4: ทดสอบในเทอร์มินัล (พิมพ์ q เพื่อออก)"""
from rag import recommend


def main():
    print("ผู้ช่วยแนะนำหนัง (พิมพ์ q เพื่อออก)")
    while True:
        query = input("\nอยากดูหนังแบบไหน: ").strip()
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue
        print("\n" + recommend(query))


if __name__ == "__main__":
    main()