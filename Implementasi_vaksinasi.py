import math
import re
import ast
import pandas as pd
from collections import Counter

# KONFIGURASI

FILEPATH          = 'hasil_text_processing_vaksinasi.xlsx'
TOP_K             = 5
MIN_PANJANG_TOKEN = 3

LABEL_SUMBER = {
    1 : 'Jurnal / Berita',
    2 : 'Media Sosial',
}

KATA_HENTI = {
    
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'adalah', 'ada', 'juga', 'tidak', 'sudah', 'pada', 'dalam', 'oleh',
    'jika', 'maka', 'agar', 'bahwa', 'namun', 'tetapi', 'akan', 'telah',
    'harus', 'saat', 'hal', 'atas', 'atas', 'atau', 'bisa', 'sama',

    'gak', 'ga', 'gue', 'gw', 'lo', 'lah', 'deh', 'sih', 'kok', 'dong',
    'tapi', 'kalo', 'kalau', 'krn', 'jd', 'bgt', 'aja', 'udah', 'sdh',
    'nggak', 'enggak', 'blm', 'karna', 'dgn', 'utk', 'yg', 'dr', 'dlm',
    'pd', 'tp', 'kl', 'sy', 'org', 'dg', 'dpt', 'jgn', 'tsb', 'bs', 'klo',

    'saya', 'aku', 'kamu', 'dia', 'kami', 'kita', 'mereka',

    'ya', 'nya', 'nih', 'tuh', 'pun', 'lagi', 'terus', 'trus',
    'memang', 'emang', 'banyak', 'semua', 'saja', 'mau', 'baru',
    'apa', 'gimana', 'banget',
}


# MEMUAT DATA

def muat_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df = df.dropna(subset=['sentence', 'text_final']).reset_index(drop=True)
    df.insert(0, 'id_dok', range(1, len(df) + 1))
    return df


# TOKENISASI

def pecah_token(teks: str) -> list[str]:
    return [t for t in str(teks).split() if len(t) >= MIN_PANJANG_TOKEN]


def bersihkan_query(teks_query: str) -> list[str]:
    teks = re.sub(r'[^a-z\s]', ' ', str(teks_query).lower())
    return [
        token for token in teks.split()
        if token not in KATA_HENTI and len(token) >= MIN_PANJANG_TOKEN
    ]


# PERHITUNGAN TF-IDF

def hitung_tf(token_list: list[str]) -> dict[str, float]:
    frekuensi = Counter(token_list)
    return {
        term: 1 + math.log10(freq)
        for term, freq in frekuensi.items()
        if freq > 0
    }


def hitung_idf(semua_token: list[list[str]]) -> dict[str, float]:
    N  = len(semua_token)
    df = {}

    for token_doc in semua_token:
        for term in set(token_doc):           
            df[term] = df.get(term, 0) + 1

    return {
        term: math.log10((N + 1) / (frek_dok + 1)) + 1
        for term, frek_dok in df.items()
    }


def hitung_tfidf(tf: dict, idf: dict) -> dict[str, float]:
    return {
        term: nilai_tf * idf.get(term, 0.0)
        for term, nilai_tf in tf.items()
    }


# VECTOR SPACE MODEL (VSM)

def cosine_similarity(vektor_a: dict, vektor_b: dict) -> float:
    semua_term  = set(vektor_a) | set(vektor_b)
    dot_product = sum(vektor_a.get(t, 0.0) * vektor_b.get(t, 0.0) for t in semua_term)
    magnitudo_a = math.sqrt(sum(v ** 2 for v in vektor_a.values()))
    magnitudo_b = math.sqrt(sum(v ** 2 for v in vektor_b.values()))

    if magnitudo_a == 0.0 or magnitudo_b == 0.0:
        return 0.0

    return dot_product / (magnitudo_a * magnitudo_b)


# MEMBANGUN INDEKS

def bangun_indeks(df: pd.DataFrame) -> tuple:
    daftar_token = [pecah_token(teks) for teks in df['text_final']]
    kamus_idf    = hitung_idf(daftar_token)
    vektor_dokumen = [
        hitung_tfidf(hitung_tf(token), kamus_idf)
        for token in daftar_token
    ]
    return daftar_token, kamus_idf, vektor_dokumen


# PENCARIAN


def cari_dokumen(query: str, vektor_dokumen: list, kamus_idf: dict,
                 top_k: int = TOP_K) -> list[tuple]:
    token_query  = bersihkan_query(query)
    if not token_query:
        return []

    vektor_query = hitung_tfidf(hitung_tf(token_query), kamus_idf)

    skor_semua = [
        (i, cosine_similarity(vektor_query, vec))
        for i, vec in enumerate(vektor_dokumen)
    ]
    skor_semua.sort(key=lambda x: x[1], reverse=True)

    return [(idx, skor) for idx, skor in skor_semua[:top_k] if skor > 0]


# UI UTAMA

import tkinter as tk
from tkinter import ttk, messagebox


def jalankan_ui(df, daftar_token, kamus_idf, vektor_dokumen):

    BG         = "#DFDEDC"
    PANEL      = "#FFFFFF"
    MERAH      = "#9E3732"
    ABU        = "#6B7280"
    ABU_MUDA   = "#E5E7EB"
    HITAM      = "#3E3439"
    BIRU       = "#607EB1"

    FONT_JUDUL  = ("Segoe UI", 16, "bold")
    FONT_LABEL  = ("Segoe UI", 10, "bold")
    FONT_NORMAL = ("Segoe UI", 10)
    FONT_KECIL  = ("Segoe UI", 9)

    root = tk.Tk()
    root.title("Mesin Pencari — Vaksinasi")
    root.geometry("960x620")
    root.configure(bg=BG)
    root.resizable(True, True)
    root.minsize(720, 460)

    frm_header = tk.Frame(root, bg=MERAH, padx=20, pady=12)
    frm_header.pack(fill="x")

    tk.Label(frm_header, text="Search Engine IR Classic",
             font=FONT_JUDUL, bg=MERAH, fg="white").pack(side="left")

    frm_input = tk.Frame(root, bg=BG, padx=20, pady=12)
    frm_input.pack(fill="x")

    tk.Label(frm_input, text="Kata Kunci :", font=FONT_LABEL,
             bg=BG, fg=HITAM).grid(row=0, column=0, sticky="w", padx=(0, 8))

    var_query = tk.StringVar()
    entry = tk.Entry(frm_input, textvariable=var_query, font=FONT_NORMAL,
                     relief="flat", bg=PANEL, fg=HITAM,
                     highlightthickness=1, highlightbackground=ABU_MUDA,
                     highlightcolor=MERAH, width=55)
    entry.grid(row=0, column=1, ipady=7, sticky="ew")

    tk.Label(frm_input, text="  Top-K :", font=FONT_LABEL,
             bg=BG, fg=HITAM).grid(row=0, column=2, padx=(12, 4))
    var_topk = tk.IntVar(value=TOP_K)
    spin = tk.Spinbox(frm_input, from_=1, to=20, textvariable=var_topk,
                      width=4, font=FONT_NORMAL, relief="flat",
                      bg=PANEL, fg=HITAM, bd=0,
                      highlightthickness=1, highlightbackground=ABU_MUDA)
    spin.grid(row=0, column=3, ipady=6)

    btn_cari = tk.Button(frm_input, text="  Cari  ", font=FONT_LABEL,
                         bg=MERAH, fg="white", relief="flat",
                         activebackground=MERAH, activeforeground="white",
                         cursor="hand2", padx=14, pady=5)
    btn_cari.grid(row=0, column=4, padx=(10, 0))

    frm_input.columnconfigure(1, weight=1)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview",         background=PANEL, foreground=HITAM,
                                        rowheight=26, fieldbackground=PANEL,
                                        font=FONT_NORMAL)
    style.configure("Treeview.Heading", background=ABU_MUDA, foreground=HITAM,
                                        font=FONT_LABEL)
    style.map("Treeview", background=[("selected", "#9E3732")])

    frm_panel = tk.Frame(root, bg=PANEL, padx=0, pady=0)
    frm_panel.pack(fill="both", expand=True, padx=20, pady=(0, 0))

    frm_info = tk.Frame(frm_panel, bg=PANEL, padx=12, pady=6)
    frm_info.pack(fill="x")
    lbl_info = tk.Label(frm_info, text="Masukkan kata kunci lalu tekan Cari.",
                        font=FONT_KECIL, bg=PANEL, fg=ABU)
    lbl_info.pack(side="left")

    kolom_hasil = ("Peringkat", "ID Dok", "Skor VSM", "Sumber", "Cuplikan Teks")
    tree_hasil  = ttk.Treeview(frm_panel, columns=kolom_hasil,
                                show="headings", selectmode="browse")
    for col, lebar in zip(kolom_hasil, [72, 68, 88, 130, 500]):
        tree_hasil.heading(col, text=col)
        tree_hasil.column(col, width=lebar, minwidth=40,
                          anchor="center" if lebar <= 130 else "w")

    sb_y = ttk.Scrollbar(frm_panel, orient="vertical",   command=tree_hasil.yview)
    tree_hasil.configure(yscrollcommand=sb_y.set)
    tree_hasil.pack(side="left", fill="both", expand=True)
    sb_y.pack(side="right",  fill="y")

    tree_hasil.tag_configure("ganjil", background="#EFF6FF")
    tree_hasil.tag_configure("genap",  background=PANEL)

    def lakukan_pencarian(event=None):
        query = var_query.get().strip()
        if not query:
            messagebox.showwarning("Perhatian", "Masukkan kata kunci terlebih dahulu.")
            return

        k     = var_topk.get()
        hasil = cari_dokumen(query, vektor_dokumen, kamus_idf, top_k=k)

        for row in tree_hasil.get_children():
            tree_hasil.delete(row)

        if not hasil:
            lbl_info.config(
                text=f'Tidak ada dokumen cocok untuk "{query}".',
                fg=MERAH
            )
            return

        token_q = bersihkan_query(query)
        lbl_info.config(
            text=(f'Query: "{query}"  →  token: {token_q}  |  '
                  f'{len(hasil)} dokumen ditemukan'),
            fg=BIRU
        )

        for rank, (idx, skor) in enumerate(hasil, 1):
            baris   = df.iloc[idx]
            sumber  = LABEL_SUMBER.get(baris.get('sumber'), '?')
            preview = str(baris['sentence'])[:100].replace('\n', ' ')
            tag     = "ganjil" if rank % 2 else "genap"
            tree_hasil.insert("", "end", iid=str(rank), tags=(tag,),
                               values=(rank, baris['id_dok'],
                                       f"{skor:.4f}", sumber, preview))

    btn_cari.config(command=lakukan_pencarian)
    entry.bind("<Return>", lakukan_pencarian)
    entry.focus()

    root.mainloop()


# PROGRAM UTAMA

def evaluasi_terminal_tfidf(df, vektor_dokumen, kamus_idf):
    import math
    queries = [
        "Penolakan vaksinasi oleh masyarakat dan kelompok antivaksin",
        "Dampak dan efektivitas vaksinasi terhadap penurunan kasus COVID-19",
        "Efek samping vaksin dan KIPI (Kejadian Ikutan Pasca Imunisasi)",
        "Diplomasi vaksin Indonesia di tingkat internasional",
        "Imunisasi anak dan vaksin campak untuk mencegah penyakit menular"
    ]
    ground_truths = [
        [8, 23, 29, 63, 69],
        [12, 27, 36, 37, 50],
        [3, 17, 40, 54, 70],
        [25, 32, 41, 42, 45],
        [52, 59, 60, 64, 74]
    ]
    
    print("\n")
    print("EVALUASI 5 GROUND TRUTH QUERY (TF-IDF)".center(60, " "))
    total_ap = 0; total_rr = 0; total_ndcg = 0
    
    for i, q in enumerate(queries):
        hasil = cari_dokumen(q, vektor_dokumen, kamus_idf, top_k=5)
        doc_ids = [df.iloc[idx]['id_dok'] for idx, skor in hasil]
        gt = ground_truths[i]
        
        ap = sum([sum([1 for x in doc_ids[:k] if x in gt])/k for k, did in enumerate(doc_ids, 1) if did in gt]) / len(gt)
        rr = next((1.0/k for k, did in enumerate(doc_ids, 1) if did in gt), 0.0)
        dcg = sum([1.0/math.log2(k+1) for k, did in enumerate(doc_ids, 1) if did in gt])
        ndcg = dcg / 2.9485
        
        total_ap += ap; total_rr += rr; total_ndcg += ndcg
        
        print(f"\nQ{i+1}: {q}")
        print(f"Top-5 Dokumen: {doc_ids}")
        print(f"Metrics -> AP: {ap:.4f} | RR: {rr:.4f} | NDCG: {ndcg:.4f}")

    print("\n")    
    print(f"RATA-RATA -> MAP: {total_ap/5:.4f} | MRR: {total_rr/5:.4f} | NDCG: {total_ndcg/5:.4f}")
    print("\n") 
    print("Membuka Antarmuka Pengguna (UI)...")

def main():
    df = muat_dataset(FILEPATH)

    daftar_token, kamus_idf, vektor_dokumen = bangun_indeks(df)

    evaluasi_terminal_tfidf(df, vektor_dokumen, kamus_idf)

    jalankan_ui(df, daftar_token, kamus_idf, vektor_dokumen)


if __name__ == '__main__':
    main()