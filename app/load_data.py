import pandas as pd
from app.chroma_setup import get_vectordb

def load_dataset_to_chroma(file_path):
    print("Memuat dataset hukum...")
    df = pd.read_excel(file_path)
    vectordb = get_vectordb()

    documents = []
    metadatas = []
    ids = []

    print("Membersihkan dan menormalisasi data...")
    for _, row in df.iterrows():
        clean_nomor_pasal = str(row['Nomor_Pasal']).strip() 
        clean_tipe = str(row["Tipe"]).strip().title() 
        clean_sumber = str(row["Sumber"]).strip().upper()
        text = f"{clean_nomor_pasal}: {row['Isi_Pasal']}"
        
        metadata = {
            "Nomor_Pasal": clean_nomor_pasal,
            "Buku": str(row["Buku"]).strip(),
            "Bab": str(row["Bab"]).strip(),
            "Judul_Bab": str(row["Judul_Bab"]).strip().title(),
            "Versi": int(row["Versi"]),
            "Jumlah_Versi": int(row["Jumlah_Versi"]),
            "Tipe": clean_tipe,
            "ID_Pasal": str(row["ID_Pasal"]),
            "Sumber": clean_sumber,
        }
        
        documents.append(text)
        metadatas.append(metadata)
        unique_id = f"{clean_sumber.replace(' ', '_')}_{row['ID_Pasal']}"
        ids.append(unique_id)

    if not documents:
        print("Tidak ada dokumen untuk diproses. Periksa file Excel.")
        return
    try:
        print(f"Menghapus koleksi '{vectordb._collection.name}' lama...")
        vectordb.delete_collection()
        print("Koleksi lama berhasil dihapus. Membuat ulang...")
        vectordb = get_vectordb() 
    except Exception as e:
        print(f"Info: Tidak dapat menghapus koleksi: {e}")

    print(f"Memasukkan {len(documents)} dokumen ke ChromaDB...")
    vectordb.add_texts(texts=documents, metadatas=metadatas, ids=ids)
    print(f"Dataset {len(df)} pasal berhasil dimasukkan ke ChromaDB.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.load_data <path_excel>")
        raise SystemExit(1)

    load_dataset_to_chroma(sys.argv[1])