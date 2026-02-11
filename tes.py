from app.chroma_setup import get_vectordb

def cut(s, n):
    s = "" if s is None else str(s)
    return (s[:n-3] + "...") if len(s) > n else s.ljust(n)

def extract_isi_pasal(doc: str):
    if not doc:
        return ""
    parts = doc.split(":", 1)
    return parts[1].strip() if len(parts) == 2 else doc

def main():
    vectordb = get_vectordb()
    data = vectordb._collection.get(limit=5, include=["documents", "metadatas"])

    headers = [
        "ID",
        "Sumber",
        "Nomor_Pasal",
        "ID_Pasal",
        "Tipe",
        "Versi",
        "Jumlah_Versi",
        "Buku",
        "Bab",
        "Judul_Bab",
        "Isi_Pasal",
    ]
    widths = [22, 10, 12, 12, 12, 6, 13, 6, 6, 22, 70]

    line = " | ".join(cut(h, w) for h, w in zip(headers, widths))
    sep  = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)

    for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        isi = extract_isi_pasal(doc)

        row = [
            id_,
            meta.get("Sumber", ""),
            meta.get("Nomor_Pasal", ""),
            meta.get("ID_Pasal", ""),
            meta.get("Tipe", ""),
            meta.get("Versi", ""),
            meta.get("Jumlah_Versi", ""),
            meta.get("Buku", ""),
            meta.get("Bab", ""),
            meta.get("Judul_Bab", ""),
            isi,
        ]
        print(" | ".join(cut(v, w) for v, w in zip(row, widths)))

if __name__ == "__main__":
    main()
