import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()

def normalize(text: str) -> str:
    # Normalisasi teks agar konsisten.
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # buang tanda baca
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Lebih enak: topic -> daftar variasi kata
# ========= KUHP + UU ITE TOPIC_TRIGGERS (Judul_Bab exact match) =========
TOPIC_TRIGGERS_KUHP = {
    # =========================
    # KUHP - Buku I (Umum)
    # =========================
    "Ruang Lingkup Berlakunya Ketentuan Peraturan Perundang-Undangan Pidana": [
        "ruang lingkup", "berlaku", "berlakunya", "asas teritorial", "asas personal",
        "asas nasional aktif", "asas nasional pasif", "asas universal",
        "tempat kejadian", "locus delicti", "waktu kejadian", "tempus delicti",
        "di luar negeri", "ekstrateritorial"
    ],
    "Tindak Pidana Dan Pertanggungjawaban Pidana": [
        "tindak pidana", "pertanggungjawaban", "kesalahan", "unsur delik",
        "dolus", "culpa", "niat", "sengaja", "lalai",
        "percobaan", "penyertaan", "turut serta", "pembantuan"
    ],
    "Pemidanaan, Pidana, Dan Tindakan": [
        "pemidanaan", "pidana", "hukuman", "sanksi", "pidana penjara",
        "kurungan", "denda", "pidana tambahan", "tindakan", "pidana percobaan"
    ],
    "Gugurnya Kewenangan Penuntutan Dan Pelaksanaan Pidana": [
        "gugur", "hapus", "daluwarsa", "kadaluarsa", "nebis in idem",
        "amnesti", "grasi", "abolisi", "meninggal", "hapus penuntutan"
    ],
    "Pengertian Istilah": [
        "pengertian", "definisi", "istilah", "yang dimaksud", "arti", "makna"
    ],
    "Aturan Penutup": [
        "aturan penutup", "penutup", "ketentuan penutup"
    ],

    # =========================
    # KUHP - Buku II (Khusus)
    # =========================
    "Tindak Pidana Terhadap Keamanan Negara": [
        "makar", "pemberontakan", "separatis", "pengkhianatan", "keamanan negara",
        "subversi", "merongrong negara"
    ],
    "Tindak Pidana Terhadap Martabat Presiden Dan/Atau Wakil Presiden": [
        "penghinaan presiden", "hinaan presiden", "fitnah presiden",
        "martabat presiden", "wakil presiden"
    ],
    "Tindak Pidana Terhadap Negara Sahabat": [
        "negara sahabat", "diplomatik", "diplomat", "kedutaan", "konsulat"
    ],
    "Tindak Pidana Terhadap Penyelenggaraan Rapat Lembaga Legislatif Dan Badan": [
        "ganggu rapat", "mengganggu rapat", "rapat dpr", "sidang", "legislatif",
        "rapat lembaga", "menghalangi rapat"
    ],
    "Tindak Pidana Terhadap Ketertiban Umum": [
        "ketertiban umum", "kerusuhan", "rusuh", "tawuran", "penghasutan",
        "menghasut", "provokasi", "kekacauan"
    ],
    "Tindak Pidana Terhadap Proses Peradilan": [
        "menghalangi penyidikan", "menghalangi peradilan", "perintangan", "obstruction",
        "menghilangkan barang bukti", "intimidasi saksi", "mengancam saksi",
        "keterangan palsu", "rekayasa perkara"
    ],
    "Tindak Pidana Terhadap Agama, Kepercayaan, Dan Kehidupan Beragama Atau": [
        "penistaan agama", "penodaan agama", "menghina agama", "penghinaan agama",
        "ibadah", "tempat ibadah"
    ],
    "Tindak Pidana Yang Membahayakan Keamanan Umum Bagi Orang, Kesehatan, Dan": [
        "membahayakan umum", "bahaya umum", "kebakaran", "ledakan", "bahan berbahaya",
        "racun", "membahayakan kesehatan", "mengganggu keselamatan"
    ],
    "Tindak Pidana Terhadap Kekuasaan Pemerintahan": [
        "melawan petugas", "melawan pemerintah", "menghalangi petugas",
        "pejabat", "kekuasaan pemerintahan"
    ],
    "Tindak Pidana Keterangan Palsu Di Atas Sumpah": [
        "sumpah palsu", "keterangan palsu", "perjury", "di atas sumpah", "bersumpah"
    ],
    "Tindak Pidana Pemalsuan Mata Uang Dan Uang Kertas": [
        "uang palsu", "memalsukan uang", "pemalsuan uang", "mata uang palsu"
    ],
    "Tindak Pidana Pemalsuan Meterai, Cap Negara, Dan Tera Negara": [
        "meterai palsu", "cap negara", "tera negara", "pemalsuan meterai"
    ],
    "Tindak Pidana Pemalsuan Surat": [
        "pemalsuan surat", "surat palsu", "memalsukan surat", "akta palsu",
        "ijazah palsu", "dokumen palsu"
    ],
    "Tindak Pidana Terhadap Asal Usul Dan Perkawinan": [
        "perkawinan", "nikah", "pernikahan", "asal usul", "status anak",
        "pencatatan nikah", "perkawinan palsu"
    ],
    "Tindak Pidana Kesusilaan": [
        "asusila", "asusilaan", "pencabulan", "cabul", "perkosa", "pemerkosaan",
        "persetubuhan", "pornografi", "eksploitasi seksual",
        "pelecehan", "pelecehan seksual", "meraba", "diraba", "rabaan", "dicolek", "groping"

    ],
    "Tindak Pidana Penelantaran Orang": [
        "penelantaran", "menelantarkan", "anak terlantar", "abaikan anak",
        "tidak memberi nafkah", "meninggalkan keluarga"
    ],
    "Tindak Pidana Penghinaan": [
        "penghinaan", "menghina", "pencemaran nama baik", "fitnah",
        "menista", "hinaan", "kata kasar", "tidak senonoh", "maki", "mencaci",
        "merendahkan", "dipermalukan"
    ],
    "Tindak Pidana Pembukaan Rahasia": [
        "pembukaan rahasia", "membuka rahasia", "bocor rahasia", "membocorkan",
        "rahasia jabatan", "rahasia perusahaan"
    ],
    "Tindak Pidana Terhadap Kemerdekaan Orang": [
        "penculikan", "culik", "penyekapan", "sekapan", "perampasan kemerdekaan",
        "menahan orang", "sandera"
    ],
    "Tindak Pidana Penyelundupan Manusia": [
        "penyelundupan manusia", "human smuggling", "migran ilegal", "selundup orang"
    ],
    "Tindak Pidana Terhadap Nyawa Dan Janin": [
        "pembunuhan", "bunuh", "menghilangkan nyawa", "pembunuhan berencana",
        "janin", "aborsi", "menggugurkan kandungan"
    ],
    "Tindak Pidana Terhadap Tubuh": [
        "penganiayaan", "aniaya", "pemukulan", "kekerasan", "melukai", "luka",
        "memar", "lebam", "cedera", "tampar", "tendang", "dorong", "mendorong", "jatuh"
    ],
    "Tindak Pidana Yang Mengakibatkan Mati Atau Luka Karena Kealpaan": [
        "kealpaan", "kelalaian", "lalai", "kecelakaan", "tabrakan",
        "mati karena lalai", "luka karena lalai"
    ],
    "Tindak Pidana Pencurian": [
        "pencurian", "mencuri", "curi", "pencuri", "nyolong", "maling",
        "jambret", "copet", "rampas"
    ],
    "Tindak Pidana Pemerasan Dan Pengancaman": [
        "pemerasan", "memeras", "pemalakan", "malak",
        "pengancaman", "mengancam", "ancaman", "blackmail",
        "dipaksa", "memaksa", "paksa", "menyerahkan", "serahkan",
        "kalau tidak kasih", "akan dipukul", "dicegat", "diadang"
    ],
    "Tindak Pidana Penggelapan": [
        "penggelapan", "menggelapkan", "gelapkan", "bawa lari", "penyalahgunaan amanah"
    ],
    "Tindak Pidana Perbuatan Curang": [
        "penipuan", "menipu", "tipu", "penipu", "curang", "fraud", "scam", "pembohongan",
        "bukti transfer palsu", "transfer palsu", "saldo tidak masuk", "marketplace", "rekening tujuan"
    ],
    "Tindak Pidana Terhadap Kepercayaan Dalam Menjalankan Usaha": [
        "kepercayaan usaha", "kecurangan usaha", "menipu dalam usaha",
        "penyalahgunaan kepercayaan", "itikad buruk", "fraud usaha"
    ],
    "Tindak Pidana Perusakan Dan Penghancuran Barang Dan Bangunan Gedung": [
        "perusakan", "merusak", "vandalisme", "penghancuran",
        "menghancurkan", "hancurkan", "merusak barang"
    ],
    "Tindak Pidana Jabatan": [
        "penyalahgunaan wewenang", "abuse of power", "suap", "gratifikasi",
        "pegawai negeri", "pejabat", "pemerasan oleh pejabat", "pungli"
    ],
    "Tindak Pidana Pelayaran": [
        "pelayaran", "kapal", "nahkoda", "pelabuhan", "awak kapal"
    ],
    "Tindak Pidana Penerbangan Dan Tindak Pidana Terhadap Sarana Serta Prasarana": [
        "penerbangan", "pesawat", "bandara", "sarana prasarana", "gangguan penerbangan"
    ],
    "Tindak Pidana Penadahan, Penerbitan, Dan Pencetakan": [
        "penadahan", "menadah", "barang curian", "tadah",
        "penerbitan", "pencetakan", "cetak"
    ],
    "Tindak Pidana Berdasarkan Hukum Yang Hidup Dalam Masyarakat": [
        "hukum yang hidup", "hukum adat", "living law", "adat"
    ],
    "Tindak Pidana Khusus": [
        "tindak pidana khusus", "pidana khusus", "khusus"
    ],
    "Ketentuan Peralihan": [
        "ketentuan peralihan", "peralihan", "transisi"
    ],
    "Ketentuan Penutup": [
        "ketentuan penutup", "penutup"
    ],
}

TOPIC_TRIGGERS_ITE = {
    # =========================
    # UU ITE
    # =========================
    "Ketentuan Umum": [
        "ketentuan umum", "pengertian", "definisi", "istilah"
    ],
    "Asas Dan Tujuan": [
        "asas", "tujuan", "prinsip", "ruang lingkup ite"
    ],
    "Informasi, Dokumen, Dan Tanda Tangan Elektronik": [
        "informasi elektronik", "dokumen elektronik", "tanda tangan elektronik",
        "ttd elektronik", "digital signature", "e-sign", "bukti elektronik"
    ],
    "Penyelenggaraan Sertifikasi Elektronik Dan Sistem Elektronik": [
        "sertifikasi elektronik", "sertifikat elektronik", "psre",
        "sistem elektronik", "pse", "penyelenggara sistem",
        "keamanan sistem", "registrasi pse"
    ],
    "Transaksi Elektronik": [
        "transaksi elektronik", "kontrak elektronik", "perjanjian elektronik",
        "e-commerce", "jual beli online", "pembayaran online"
    ],
    "Nama Domain, Hak Kekayaan Intelektual, Dan Perlindungan Hak Pribadi": [
        "nama domain", "domain", "hak kekayaan intelektual", "hki",
        "hak cipta", "merek", "privasi", "hak pribadi", "data pribadi", "doxing"
    ],
    "Perbuatan Yang Dilarang": [

        "akses ilegal", "akses tanpa izin", "hacking", "intersepsi", "penyadapan",
        "dibobol", "diretas", "peretasan", "password diganti", "reset password",
        "ambil alih akun", "akun dibajak", "akun dibobol", "login tanpa izin",
        "masuk akun orang lain", "email dibobol", "gmail dibobol", "surat elektronik",
        "gangguan sistem", "ddos", "malware", "virus", "ransomware",
        "phishing", "carding", "manipulasi data", "ubah data", "hapus data",
        "akun palsu", "akun fake", "impersonasi", "menyamar", "mengatasnamakan",
        "identitas palsu", "profil palsu", "seolah olah", "seolah-olah",

        "penghinaan", "menghina", "hinaan",
        "pencemaran nama baik", "mencemarkan", "nama baik",
        "fitnah", "menuduh", "tuduhan", "tanpa bukti",
        "hoaks", "berita bohong", "ujaran kebencian",
        "ancaman", "mengancam", "pemerasan", "memeras",
        "menakut-nakuti", "menakuti", "ancaman kekerasan",
        "whatsapp", "wa", "dm", "chat", "pesan langsung", "langsung ke korban",
        "telegram", "line", "messenger", "discord", "facebook", "fb", "x", "twitter",
        "media sosial", "sosmed", "postingan", "unggahan", "voice note", "pesan suara", "vn",
        "pasal 29", "pasal 27b",
    ],
    "Penyelesaian Sengketa": [
        "sengketa", "ganti rugi", "tuntutan", "mediasi", "arbitrase",
        "penyelesaian sengketa"
    ],
    "Peran Pemerintah Dan Peran Masyarakat": [
        "peran pemerintah", "peran masyarakat", "kominfo", "pengawasan",
        "pembinaan", "edukasi", "literasi digital"
    ],
    "Penyidikan": [
        "penyidikan", "penyidik", "alat bukti elektronik",
        "penggeledahan", "penyitaan", "penangkapan", "forensik digital"
    ],
    "Ketentuan Pidana": [
        "ketentuan pidana", "sanksi uu ite", "ancaman uu ite", "pidana uu ite",
    ],
    "Ketentuan Peralihan": [
        "ketentuan peralihan", "peralihan", "transisi"
    ],
    "Ketentuan Penutup": [
        "ketentuan penutup", "penutup"
    ],
}

STEM_TRIGGERS_KUHP = {
    topic: {stemmer.stem(t) for t in triggers}
    for topic, triggers in TOPIC_TRIGGERS_KUHP.items()
}

STEM_TRIGGERS_ITE = {
    topic: {stemmer.stem(t) for t in triggers}
    for topic, triggers in TOPIC_TRIGGERS_ITE.items()
}

ITE_EXPLICIT_SIGNAL_RE = re.compile(
    r"\b(uu\s*ite|informasi elektronik|dokumen elektronik|sistem elektronik|"
    r"mendistribusikan|mentransmisikan|tanpa hak|intersepsi|akses ilegal|hacking|"
    r"penyadapan|viralkan|doxing|chat|dm|whatsapp|wa|instagram|ig|tiktok|"
    r"telegram|line|messenger|discord|facebook|fb|x|twitter|media sosial|sosmed|postingan|unggahan|"
    r"email|surel|akun|password|login|dibobol|diretas|peretasan|"
    r"akun palsu|impersonasi|mengatasnamakan|seolah-olah)\b",
    re.I
)

ITE_SUBSTANTIVE_SIGNAL_RE = re.compile(
    r"\b(uu\s*ite|informasi elektronik|dokumen elektronik|sistem elektronik|"
    r"mendistribusikan|mentransmisikan|tanpa hak|intersepsi|akses ilegal|hacking|"
    r"penyadapan|viralkan|doxing|media sosial|sosmed|postingan|unggahan|"
    r"email|surel|akun|password|login|"
    r"dibobol|diretas|peretasan|akun palsu|impersonasi|mengatasnamakan|seolah-olah)\b",
    re.I
)
ITE_CHANNEL_SIGNAL_RE = re.compile(
    r"\b(chat|dm|whatsapp|wa|instagram|ig|tiktok|telegram|line|messenger|discord|"
    r"facebook|fb|x|twitter|email|surel|gmail|voice note|pesan suara|vn)\b",
    re.I,
)
ITE_CHANNEL_CRIME_CONTEXT_RE = re.compile(
    r"\b(ancam|mengancam|ancaman|pemeras|memeras|minta transfer|transfer|"
    r"hajar|bunuh|habisin|babak belur|tusuk|gebuk|gebukin|bacok|menakut|takut|kekerasan|"
    r"fitnah|menghina|pencemaran|menuduh|tuduh|nama baik|reputasi|postingan|unggahan|"
    r"sebar|viralkan|akun palsu|impersonasi|dibobol|diretas|peretasan|"
    r"password|login|akses tanpa izin|intersepsi|penyadapan)\b",
    re.I
)
ITE_MESSAGE_DELIVERY_RE = re.compile(
    r"\b(lewat|via|melalui|kirim|mengirim|dikirim)\b.*\b(whatsapp|wa|dm|chat|telegram|line|messenger|discord|"
    r"facebook|fb|x|twitter|pesan|email|surel|gmail|voice note|pesan suara|vn)\b|"
    r"\b(whatsapp|wa|dm|chat|telegram|line|messenger|discord|facebook|fb|x|twitter|"
    r"pesan|email|surel|gmail|voice note|pesan suara|vn)\b.*\b(kirim|mengirim|dikirim|langsung|personal|kepada saya|kepada korban)\b",
    re.I,
)
KUHP_PHYSICAL_STRONG_RE = re.compile(
    r"\b(luka berat|berdarah|patah|tulang|cacat|mati|tewas|meninggal)\b|"
    r"\b(memukul|dipukul|pukul|mendorong|dorong|tampar|tendang|hantam|menyerang)\b.*\bsampai\b",
    re.I
)

KUHP_CORE_SIGNAL_RE = re.compile(
    r"\b(pencurian|mencuri|curi|penggelapan|menggelapkan|penipuan|menipu|tipu|"
    r"pemerasan|memeras|pengancaman|mengancam|penganiayaan|aniaya|"
    r"pelecehan|pencabulan|cabul|perkosa|persetubuhan|penghinaan|fitnah|"
    r"pukul|memukul|dipukul|dorong|mendorong|tampar|tendang|luka|memar|berdarah|patah|tulang|cacat)\b",
    re.I
)

ITE_ONLY_TOPICS = {
    "Informasi, Dokumen, Dan Tanda Tangan Elektronik",
    "Penyelenggaraan Sertifikasi Elektronik Dan Sistem Elektronik",
    "Transaksi Elektronik",
    "Nama Domain, Hak Kekayaan Intelektual, Dan Perlindungan Hak Pribadi",
    "Perbuatan Yang Dilarang",
    "Penyelesaian Sengketa",
    "Peran Pemerintah Dan Peran Masyarakat",
    "Penyidikan",
}

KUHP_GENERIC_TOPICS = {
    "Ruang Lingkup Berlakunya Ketentuan Peraturan Perundang-Undangan Pidana",
    "Tindak Pidana Dan Pertanggungjawaban Pidana",
    "Pemidanaan, Pidana, Dan Tindakan",
    "Gugurnya Kewenangan Penuntutan Dan Pelaksanaan Pidana",
    "Pengertian Istilah",
    "Aturan Penutup",
    "Ketentuan Peralihan",
    "Ketentuan Penutup",
}

def detect_topics(query: str, sumber: str | None) -> list[str]:
    # Deteksi topik hukum dari pertanyaan.
    q = normalize(query)
    stems = [stemmer.stem(t) for t in q.split()]

    grams = set(stems)
    for n in (2, 3):
        grams |= {" ".join(stems[i:i+n]) for i in range(len(stems)-n+1)}

    hits = []
    if sumber in ("ITE", "UU ITE"):
        stem_triggers = STEM_TRIGGERS_ITE
        hits = [topic for topic, trig in stem_triggers.items() if grams & trig]
    elif sumber == "KUHP":
        stem_triggers = STEM_TRIGGERS_KUHP
        hits = [topic for topic, trig in stem_triggers.items() if grams & trig]
    else:
        # sumber belum ditentukan: hit dari KUHP + ITE tanpa saling override key
        for topic, trig in STEM_TRIGGERS_KUHP.items():
            if grams & trig and topic not in hits:
                hits.append(topic)
        for topic, trig in STEM_TRIGGERS_ITE.items():
            if grams & trig and topic not in hits:
                hits.append(topic)
    return hits

def hint_ite_pasal(query: str) -> str | None:
    # Berikan hint pasal ITE dari kata kunci.
    q = normalize(query)

    impersonation_words = re.search(
        r"(akun palsu|akun fake|akun bodong|impersonasi|menyamar|mengatasnamakan|"
        r"identitas palsu|profil palsu|seolah olah|seolah-olah|"
        r"pakai foto|memakai foto|pakai nama|memakai nama|foto saya|nama saya)",
        q
    )
    manipulation_words = re.search(
        r"(manipulasi|penciptaan|perubahan|penghilangan|pengrusakan|"
        r"informasi elektronik palsu|dokumen elektronik palsu|agar dianggap otentik)",
        q
    )
    penghinaan_words = re.search(
        r"(pencemaran|nama baik|fitnah|menuduh|tuduh|menghina|hinaan|reputasi|kehormatan)",
        q
    )
    violent_threat_words = re.search(
        r"(hajar|bunuh|habisin|habisi|babak belur|tusuk|gebuk|gebukin|bacok|ancaman kekerasan|menakut)",
        q
    )
    threat_words = re.search(
        r"(ancam|mengancam|ancaman|hajar|bunuh|habisin|habisi|babak belur|tusuk|gebuk|gebukin|bacok|"
        r"menakut|takut|kekerasan)",
        q
    )
    negated_threat = re.search(r"(tanpa|tidak ada)\s+ancaman(\s+kekerasan)?", q)
    direct_words = re.search(
        r"(whatsapp|wa|dm|chat|telegram|line|messenger|discord|facebook|fb|x|twitter|"
        r"voice note|pesan suara|vn|pesan langsung|langsung ke|langsung kepada|"
        r"kepada saya|kepada korban|secara personal|korban)",
        q
    )
    extortion_words = re.search(
        r"(minta uang|minta transfer|transfer|tebusan|bayar|pemeras|memeras|"
        r"menguntungkan diri|sebar|viralkan|foto intim|video intim|aib)",
        q
    )
    akses_ilegal_words = re.search(
        r"(dibobol|diretas|peretasan|password diganti|reset password|"
        r"login tanpa izin|masuk akun orang lain|akses tanpa izin|"
        r"akun email|email|gmail|kotak surat)",
        q
    )
    intersepsi_words = re.search(
        r"(intersepsi|penyadapan|menyadap|membaca pesan|membaca email|meneruskan isi email)",
        q
    )

    if negated_threat:
        threat_words = None
        violent_threat_words = None

    # Sextortion: ancam sebar konten intim + minta uang/transfer
    if (("foto" in q or "video" in q or "konten" in q)
        and ("intim" in q or "telanjang" in q or "seks" in q)
        and ("ancam" in q or "mengancam" in q or "sebar" in q or "viralkan" in q)
        and ("uang" in q or "transfer" in q or "minta" in q)):
        return "Pasal 27B"

    # Ancaman + pemerasan via elektronik -> Pasal 27B
    if threat_words and extortion_words:
        return "Pasal 27B"

    # Ancaman kekerasan/menakut-nakuti yang dikirim langsung ke korban -> Pasal 29
    if (violent_threat_words and direct_words) or (threat_words and direct_words and not penghinaan_words):
        return "Pasal 29"

    # Ancaman kekerasan via sarana elektronik tanpa indikator pemerasan.
    if violent_threat_words and not extortion_words:
        return "Pasal 29"

    # Penyamaran/manipulasi akun atau identitas digital -> Pasal 35
    if impersonation_words or manipulation_words:
        return "Pasal 35"

    # Akses tanpa hak ke akun/sistem elektronik -> Pasal 30
    if akses_ilegal_words:
        return "Pasal 30"

    # Intersepsi/penyadapan isi komunikasi elektronik -> Pasal 31
    if intersepsi_words:
        return "Pasal 31"

    if re.search(r"(pencemaran|nama baik|fitnah|menuduh|tuduh|menghina|hinaan)", q):
        return "Pasal 27A"

    return None








def parse_query(query: str):
    """
    Fungsi untuk mengekstrak informasi penting dari pertanyaan hukum.
    Mengembalikan tuple: (Nomor_Pasal, Sumber, Versi, Tipe, List_Topik_Hukum)
    """
    nomor_pasal = None
    sumber = None
    versi = 1
    tipe = "Batang Tubuh"
    
    # 🚨 Perubahan utama: Gunakan List untuk topik_hukum
    topik_hukum_list = []
    
    query_lower = query.lower()

    # Suffix pasal hanya 1 huruf (mis. 27A). Jangan salah baca "UU" sebagai suffix.
    match = re.search(
        r"\bpasal\s*(\d+)(?:\s*([a-z])(?![a-z]))?\b",
        query_lower,
        re.IGNORECASE,
    )

    minta_pasal_lain = bool(re.search(r"(pasal lain|selain uu ite|selain ite|selain kuhp)", query_lower))

    if match:
        nomor = match.group(1)
        suf  = (match.group(2) or "").upper()  # A/B dst
        nomor_pasal = f"Pasal {nomor}{suf}"


    has_kuhp = bool(re.search(r"\bkuhp\b", query_lower))
    has_ite  = bool(re.search(r"\buu\s*ite\b|\bite\b", query_lower))
    strong_physical = bool(KUHP_PHYSICAL_STRONG_RE.search(query_lower))

    q = query_lower or ""
    if ITE_SUBSTANTIVE_SIGNAL_RE.search(q):
        explicit_ite = True
    elif ITE_CHANNEL_SIGNAL_RE.search(q) and ITE_CHANNEL_CRIME_CONTEXT_RE.search(q):
        # "hajar/babak belur" di konteks pesan WA/DM tetap ITE karena itu ancaman yang dikirim.
        if KUHP_PHYSICAL_STRONG_RE.search(q) and not ITE_MESSAGE_DELIVERY_RE.search(q):
            explicit_ite = False
        else:
            explicit_ite = True
    else:
        explicit_ite = False

    core_kuhp = bool(KUHP_CORE_SIGNAL_RE.search(q))

    if has_kuhp and has_ite:
        sumber = None
        minta_pasal_lain = True
    elif has_kuhp:
        sumber = "KUHP"
    elif has_ite:
        sumber = "ITE"
    else:
        # fallback sumber berbasis sinyal fakta, agar tidak mudah tercampur lintas UU
        if explicit_ite and (not core_kuhp):
            sumber = "ITE"
        elif explicit_ite and ITE_CHANNEL_SIGNAL_RE.search(query_lower):
            sumber = "ITE"
        elif core_kuhp and strong_physical:
            sumber = "KUHP"
        elif core_kuhp and not explicit_ite:
            sumber = "KUHP"


    if "penjelasan" in query_lower or "lanjutan" in query_lower:
        versi = 2
        tipe = "Penjelasan"
    else:
        versi = 1
        tipe = "Batang Tubuh"


        # kalau ITE dan user tidak sebut pasal, coba "hint" pasal yang tepat
    if sumber == "ITE" and not nomor_pasal:
        hinted = hint_ite_pasal(query_lower)
        if hinted:
            nomor_pasal = hinted


    if not nomor_pasal:
        topik_hukum_list = detect_topics(query, sumber)

    if sumber is None and (not explicit_ite):
        # tanpa sinyal elektronik eksplisit, hindari topik ITE agar retrieval tidak nyasar
        topik_hukum_list = [t for t in topik_hukum_list if t not in ITE_ONLY_TOPICS]
    elif sumber is None and strong_physical and (not has_ite):
        # kata "chat/WA" bisa muncul sebagai bukti rencana, bukan modus ITE.
        # untuk kekerasan fisik kuat, hindari topik ITE jika user tidak eksplisit minta ITE.
        topik_hukum_list = [t for t in topik_hukum_list if t not in ITE_ONLY_TOPICS]

    # Jika user menegaskan tidak ada ancaman/kekerasan dan tidak ada fakta fisik,
    # jangan dorong topik "Terhadap Tubuh" hanya karena kata "kekerasan" muncul sebagai negasi.
    negated_violence = bool(re.search(r"\b(tidak ada|tanpa)\s+ancaman(?:\s+kekerasan)?\b", query_lower))
    has_physical_fact = bool(re.search(r"\b(pukul|memukul|dorong|mendorong|tampar|tendang|luka|memar|cedera|berdarah)\b", query_lower))
    if negated_violence and (not has_physical_fact):
        topik_hukum_list = [t for t in topik_hukum_list if t != "Tindak Pidana Terhadap Tubuh"]

    # "reputasi rusak / nama baik rusak" sering salah terbaca sebagai perusakan barang.
    if re.search(r"\b(reputasi|nama baik|kehormatan|pencemaran|fitnah|penghinaan)\b", query_lower):
        topik_hukum_list = [t for t in topik_hukum_list if t != "Tindak Pidana Perusakan Dan Penghancuran Barang Dan Bangunan Gedung"]

    # Jika sudah ada topik KUHP yang spesifik, buang topik KUHP umum agar retrieval tidak noise.
    if topik_hukum_list:
        has_specific_kuhp = any((t not in KUHP_GENERIC_TOPICS) and (t not in ITE_ONLY_TOPICS) for t in topik_hukum_list)
        if has_specific_kuhp:
            topik_hukum_list = [t for t in topik_hukum_list if t not in KUHP_GENERIC_TOPICS]

    # Frasa "bukan menuduh ..." sering membawa kata "mencuri" sebagai contoh,
    # bukan inti delik pencurian.
    negated_tuduh = bool(re.search(r"\b(bukan|tidak|tanpa)\s+menuduh\b", query_lower))
    insult_context = bool(re.search(r"\b(penghinaan|pencemaran|nama baik|fitnah|kata kasar|tidak senonoh|maki|mencaci|dipermalukan)\b", query_lower))
    if negated_tuduh and insult_context:
        topik_hukum_list = [t for t in topik_hukum_list if t != "Tindak Pidana Pencurian"]
        
    if sumber == "ITE" and re.search(r"\b(ancaman|pidana|denda|penjara)\b", query_lower):
        if topik_hukum_list and "Ketentuan Pidana" not in topik_hukum_list:
            topik_hukum_list.append("Ketentuan Pidana")


    return nomor_pasal, sumber, versi, tipe, topik_hukum_list, minta_pasal_lain
