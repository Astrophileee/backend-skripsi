# Chatbot Hukum RAG Backend

Backend FastAPI untuk tanya jawab hukum berbasis RAG dengan ChromaDB, embedding Hugging Face, dan LLM lokal via Ollama.

## Stack utama

- FastAPI
- ChromaDB
- LangChain
- sentence-transformers
- Ollama
- Ragas untuk evaluasi

## Versi Python

Project ini diuji pada Python `3.10.6`. Disarankan memakai Python `3.10.x` agar dependency lebih stabil.

## Isi repo yang penting

- `main.py`: entry point FastAPI
- `app/`: logika API, query parsing, retriever, dan pipeline RAG
- `chroma_db/`: database vector Chroma yang saat ini ikut tersimpan di repo
- `Pasal_UU_KUHP_ITE_Combined.xlsx`: dataset sumber untuk rebuild ChromaDB
- `pengujian/`: script evaluasi dan cek model
- `.env.example`: contoh env untuk script evaluasi

## Clone dan install

### 1. Clone repository

```powershell
git clone <url-repository>
cd backend_skripsi
```

### 2. Buat virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependency Python

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Setup Ollama

Backend ini memakai model lokal `llama3:instruct` di `app/rag_pipeline.py`, jadi Ollama harus terpasang dan modelnya harus tersedia.

### 1. Install Ollama

Install Ollama di laptop yang akan menjalankan backend.

### 2. Download model yang dipakai project

```powershell
ollama pull llama3:instruct
```

### 3. Pastikan service Ollama aktif

Jika Ollama belum berjalan, jalankan service atau buka aplikasinya terlebih dahulu.

## Menjalankan API

```powershell
uvicorn main:app --reload
```

Jika berhasil, endpoint utama akan tersedia di:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

Contoh request:

```text
GET /ask?pertanyaan=orang+yang+mengambil+motor+tanpa+izin+kena+pasal+berapa
```

## Tentang `.env` dan `.gitignore`

File yang tidak ikut ke GitHub karena `.gitignore`:

- `.env`
- `venv/`
- `.venv/`
- `__pycache__/`
- `*.pyc`

Artinya setelah clone:

- harus membuat virtual environment sendiri
- install dependency sendiri
- file `.env` tidak ikut, jadi kalau ingin menjalankan evaluasi harus membuat `.env` dari `.env.example`

Catatan: folder `chroma_db/` saat ini masih ikut ter-track di Git, jadi hasil clone biasanya langsung membawa vector database yang sudah ada.

## Rebuild ChromaDB

Kalau `chroma_db/` tidak ada, rusak, atau ingin diisi ulang dari file Excel:

```powershell
python -m app.load_data Pasal_UU_KUHP_ITE_Combined.xlsx
```

Opsional untuk cek isi data awal:

```powershell
python tes.py
```

## Menjalankan evaluasi Ragas

Script evaluasi ada di folder `pengujian/` dan membutuhkan API key Gemini.

### 1. Buat file env

```powershell
Copy-Item .env.example .env
```

### 2. Isi minimal variabel berikut di `.env`

```env
GEMINI_API_KEY=isi_api_key_anda
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.5-flash-lite
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Cek model yang tersedia

```powershell
python pengujian/list_model.py
```

### 4. Jalankan evaluasi

```powershell
python pengujian/ragas.py
```

## Troubleshooting singkat

### `ModuleNotFoundError`

Pastikan virtual environment aktif, lalu jalankan ulang:

```powershell
pip install -r requirements.txt
```

### Error koneksi Ollama

Pastikan:

- Ollama sudah terinstall
- model `llama3:instruct` sudah di-pull
- service Ollama sedang aktif

### Error saat baca file Excel

Dependency `openpyxl` sudah ada di `requirements.txt`. Jika masih error, install ulang dependency:

```powershell
pip install -r requirements.txt
```

### Jawaban kosong atau retrieval tidak sesuai

Cek apakah `chroma_db/` ada. Jika tidak ada atau datanya ingin diperbarui, rebuild ulang:

```powershell
python -m app.load_data Pasal_UU_KUHP_ITE_Combined.xlsx
```
